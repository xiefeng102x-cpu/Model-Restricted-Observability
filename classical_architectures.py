"""
classical_architectures.py
====================================
Shared, reusable classical architectures for the escalating-dequantization-
attempt ladder (candidate 1: real MNIST comparison; candidate 2: synthetic
multi-n scaling sweep). Both candidates import from this module rather than
duplicating architecture code, so the SAME implementation is used everywhere
the "how quantum-like does a classical system need to be" question is asked.

Escalation levels beyond the existing Unconstrained MLP / Bloch-ball MLP
(already have real numbers for these from prior runs, not reimplemented here):

  Level 2 -- OrthogonalMLP: weight matrices constrained to be exactly
    orthogonal via matrix-exponential-of-skew-symmetric parametrization
    (W = expm(A - A^T), a standard, exact parametrization of SO(n) -- not an
    approximate regularizer). Tests whether norm-preservation / orthogonality
    as an ABSTRACT property (regardless of matching the circuit's specific
    tensor/entangling structure) is enough to resist Mahalanobis collapse.

  Level 3 -- RotationCascadeNet: NOT a generic network -- reuses the exact
    same tensor/reshape/entangling-permutation code as the quantum circuit
    (topology_mechanism_sweep.py's simulate_final_state), but with
    the RZ (phase) gate removed and restricted to REAL-valued amplitude
    vectors throughout. This isolates a specific question: is COMPLEX PHASE
    (a resource with no classical analog) the actual load-bearing ingredient,
    or does the same abstract structure (local rotations + fixed entangling
    permutation pattern) already suffice once mimicked exactly, even in reals?

    Consequence (not a bug, a structural fact worth reporting honestly): a
    real-only system cannot compute a meaningful "Y" analog -- the Y
    measurement basis change (H-then-S-dagger) inherently requires a complex
    phase factor (i). So RotationCascadeNet's natural feature set is
    "XZ" (2n-dim), not "XYZ" (3n-dim) -- this maps directly onto this
    project's own existing "measurement ladder" framework (Z-only / XZ /
    XYZ tiers), not an ad hoc omission.

Self-tests:
  1. OrthogonalMLP: W^T @ W ≈ I for random parameters (exact orthogonality).
  2. RotationCascadeNet: mathematically identical to the general complex
     circuit (simulate_final_state) when RZ angles are fixed to 0 and the
     input is real -- verified by direct numerical comparison against
     topology_mechanism_sweep.py's own validated circuit code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ========================================================================= Level 2: OrthogonalMLP

class OrthogonalLinear(nn.Module):
    """A linear layer whose weight matrix is EXACTLY orthogonal via
    W = matrix_exp(A - A^T) for a free (learnable) matrix A. This is an
    exact parametrization of SO(n) (up to the connected component), not an
    approximate penalty -- W^T @ W = I holds to numerical precision for any A."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.A = nn.Parameter(torch.randn(dim, dim) * 0.1)

    def weight(self):
        skew = self.A - self.A.T
        return torch.linalg.matrix_exp(skew)

    def forward(self, x):
        return x @ self.weight().T


class OrthogonalMLP(nn.Module):
    """input_dim -> [OrthogonalLinear(input_dim) once, then project to
    3*n_qubits via a final linear readout, BALL-PROJECTED PER TRIPLET exactly
    like the Level-1 Bloch baseline]. The orthogonal layers preserve norm
    exactly at every step (mimicking unitary evolution's norm-preservation).

    IMPORTANT: the final readout is ball-projected (same as BlochHead) so
    that boundedness is held CONSTANT between Level 1 (Bloch) and Level 2
    (Orthogonal) -- orthogonality of the hidden transform is the only new
    variable being tested here. An earlier version left the final readout
    unconstrained, which let Phase-2's weight attack drive output magnitude
    unboundedly large (observed baseline Mahalanobis ~189 vs ~30-80 for
    every bounded architecture) and eventually overflowed matrix_exp into
    NaN -- a genuine confound (boundedness change, not just orthogonality),
    not merely a numerical-stability nuisance.

    n_layers orthogonal blocks are chained (analogous to circuit depth)."""
    def __init__(self, input_dim: int, n_qubits: int, n_layers: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([OrthogonalLinear(input_dim) for _ in range(n_layers)])
        self.activation = nn.Tanh()
        self.readout = nn.Linear(input_dim, 3 * n_qubits)
        self.n_qubits = n_qubits

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = self.activation(block(h))
        z = self.readout(h)
        z3 = z.view(-1, self.n_qubits, 3)
        norms = z3.norm(dim=-1, keepdim=True).clamp(min=1.0)
        z3 = z3 / norms
        return torch.cat([z3[:, :, 0], z3[:, :, 1], z3[:, :, 2]], dim=1)


# ========================================================================= Level 3: RotationCascadeNet (real-only, structure-matched)

def _real_ry_matrix(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def _apply_single_unit_real(state, gate, unit, n_units):
    """Identical tensor/reshape mechanics to the quantum circuit's
    apply_single_qubit_gate, but state is REAL-valued (float64, not complex)."""
    axis = unit + 1
    state = torch.movedim(state, axis, 1)
    shape = state.shape
    state = state.reshape(shape[0], 2, -1)
    state = torch.einsum("ij,bjk->bik", gate.to(state.dtype), state)
    state = state.reshape(shape)
    state = torch.movedim(state, 1, axis)
    return state


def _apply_cnot_real(state, control, target, n_units):
    """Identical to the quantum circuit's apply_cnot -- CNOT is a basis-state
    PERMUTATION regardless of whether amplitudes are real or complex, so this
    code is byte-for-byte reusable."""
    c_axis, t_axis = control + 1, target + 1
    state = torch.movedim(state, (c_axis, t_axis), (1, 2))
    s0 = state[:, 0, :].clone()
    s1 = state[:, 1, :].clone()
    new_s1 = torch.flip(s1, dims=[1])
    state = torch.stack([s0, new_s1], dim=1)
    state = torch.movedim(state, (1, 2), (c_axis, t_axis))
    return state


def simulate_real_rotation_cascade(x_batch, weights, n_units, n_layers, entangling_pairs):
    """Real-valued analog of simulate_final_state: SAME tensor/entangling
    structure, RZ (phase) gate removed entirely, real amplitude-normalized
    input instead of complex. weights: (n_layers * n_units,) -- one RY angle
    per unit per layer (half as many params as the quantum circuit, since
    there is no RZ angle)."""
    B = x_batch.shape[0]
    norm = x_batch.norm(dim=-1, keepdim=True) + 1e-12
    amp = (x_batch / norm).to(torch.float64)
    state = amp.reshape((B,) + (2,) * n_units)
    wr = weights.reshape(n_layers, n_units)
    for l in range(n_layers):
        for q in range(n_units):
            g_ry = _real_ry_matrix(wr[l, q])
            state = _apply_single_unit_real(state, g_ry, q, n_units)
        for (c, t) in entangling_pairs:
            state = _apply_cnot_real(state, c, t, n_units)
    return state


_H_GATE_REAL = torch.tensor([[1, 1], [1, -1]], dtype=torch.float64) / np.sqrt(2)


def xz_features_from_real_state(state, n_units):
    """Real-only analog of xyz_features_from_state: only X and Z are
    computable without complex phase (Y requires an explicit i factor via
    H-then-S-dagger, which has no real-valued form). Returns (N, 2*n_units)
    in order [X_0..X_n, Z_0..Z_n]."""
    B = state.shape[0]
    flat = state.reshape(B, 2 ** n_units)
    probs = flat ** 2  # real amplitudes -> probabilities without conjugation
    idx = torch.arange(2 ** n_units)
    bits = ((idx.unsqueeze(1) >> torch.arange(n_units - 1, -1, -1)) & 1)
    z_ev = (1.0 - 2.0 * bits.to(probs.dtype)).T
    z_vals = probs @ z_ev.T

    x_vals_list = []
    for q in range(n_units):
        sx = _apply_single_unit_real(state, _H_GATE_REAL, q, n_units).reshape(B, 2 ** n_units)
        px = sx ** 2
        x_vals_list.append(px @ z_ev[q])
    x_vals = torch.stack(x_vals_list, dim=1)
    return torch.cat([x_vals, z_vals], dim=1)


# ========================================================================= Level 3b: PermutationCascadeNet (entangling structure WITHOUT norm-preservation)

def _generic_2x2_matrix(raw4):
    """4 free real params -> a generic (non-orthogonal) 2x2 matrix, no constraint."""
    return raw4.reshape(2, 2)


def simulate_permutation_cascade(x_batch, weights, n_units, n_layers, entangling_pairs):
    """SAME entangling-permutation structure as simulate_real_rotation_cascade,
    but each per-unit local transform is a GENERIC unconstrained 2x2 matrix
    (4 free params) instead of a norm-preserving rotation (1 param), followed
    by tanh (matching the stabilizing nonlinearity Bloch/OrthogonalMLP already
    use between layers -- kept identical across architectures so "boundedness
    via activation" is not itself a confound). Isolates: does the entangling-
    permutation structure alone (WITHOUT norm-preservation) already give
    Mahalanobis-collapse-resistance, completing the 2x2 factorial together
    with Bloch (no,no), OrthogonalMLP (yes,no), RotationCascadeNet (yes,yes):
        norm-preserving x entangling-structure
      no,  no  -> Bloch
      yes, no  -> OrthogonalMLP
      no,  yes -> PermutationCascadeNet  (this one)
      yes, yes -> RotationCascadeNet
    weights: (n_layers, n_units, 2, 2)."""
    B = x_batch.shape[0]
    norm = x_batch.norm(dim=-1, keepdim=True) + 1e-12
    amp = (x_batch / norm).to(torch.float64)
    state = amp.reshape((B,) + (2,) * n_units)
    for l in range(n_layers):
        for q in range(n_units):
            g = _generic_2x2_matrix(weights[l, q])
            state = _apply_single_unit_real(state, g, q, n_units)
            state = torch.tanh(state)
        for (c, t) in entangling_pairs:
            state = _apply_cnot_real(state, c, t, n_units)
    return state


class PermutationCascadeNet(nn.Module):
    """Wraps simulate_permutation_cascade. The state is renormalized to unit
    norm ONLY at the final readout (xz_features_from_real_state requires a
    normalized vector to interpret flat**2 as valid Born-rule-style
    probabilities) -- internal dynamics have NO norm constraint, which is the
    actual variable under test. This keeps the readout FORMULA identical to
    RotationCascadeNet's, isolating internal norm-preservation as the only
    difference between the two."""
    def __init__(self, n_units: int, n_layers: int, entangling_pairs: list[tuple[int, int]]):
        super().__init__()
        self.n_units = n_units
        self.n_layers = n_layers
        self.entangling_pairs = entangling_pairs
        self.weights = nn.Parameter(torch.randn(n_layers, n_units, 2, 2, dtype=torch.float64) * 0.1
                                     + torch.eye(2, dtype=torch.float64) * 0.9)

    def forward(self, x):
        state = simulate_permutation_cascade(x.to(torch.float64), self.weights, self.n_units, self.n_layers, self.entangling_pairs)
        flat = state.reshape(state.shape[0], -1)
        state_normalized = (flat / (flat.norm(dim=-1, keepdim=True) + 1e-12)).reshape(state.shape)
        return xz_features_from_real_state(state_normalized, self.n_units).to(torch.float32)


class RotationCascadeNet(nn.Module):
    """Wraps simulate_real_rotation_cascade + xz_features as an nn.Module
    with a trainable weight vector, matching the interface style of the
    other classical architectures (forward returns (N, 2*n_units) XZ features)."""
    def __init__(self, n_units: int, n_layers: int, entangling_pairs: list[tuple[int, int]]):
        super().__init__()
        self.n_units = n_units
        self.n_layers = n_layers
        self.entangling_pairs = entangling_pairs
        self.weights = nn.Parameter(torch.randn(n_layers * n_units, dtype=torch.float64) * 0.1)

    def forward(self, x):
        state = simulate_real_rotation_cascade(x.to(torch.float64), self.weights, self.n_units, self.n_layers, self.entangling_pairs)
        return xz_features_from_real_state(state, self.n_units).to(torch.float32)


# ========================================================================= self-test

def self_test():
    print("Running self-test (classical_architectures.py)...")

    torch.manual_seed(0)
    ortho = OrthogonalLinear(8)
    W = ortho.weight()
    identity_check = (W.T @ W - torch.eye(8)).abs().max().item()
    assert identity_check < 1e-5, f"OrthogonalLinear not orthogonal: max|W^T W - I| = {identity_check}"
    print(f"  [check] OrthogonalLinear: max|W^T W - I| = {identity_check:.2e} (expect ~0)")

    # RotationCascadeNet must be mathematically identical to the general complex
    # quantum circuit (topology_mechanism_sweep.py) when RZ=0 and input is real.
    import importlib
    tms = importlib.import_module("topology_mechanism_sweep")

    n_units, n_layers = 4, 3
    pairs = [(q, (q + 1) % n_units) for q in range(n_units)]
    torch.manual_seed(1)
    x_np = np.random.RandomState(0).randn(3, 2 ** n_units).astype(np.float64)
    ry_angles = np.random.RandomState(1).randn(n_layers * n_units).astype(np.float64) * 0.5

    x_t = torch.tensor(x_np, dtype=torch.float64)
    real_weights = torch.tensor(ry_angles, dtype=torch.float64)
    state_real = simulate_real_rotation_cascade(x_t, real_weights, n_units, n_layers, pairs)
    feats_real = xz_features_from_real_state(state_real, n_units)

    # General complex circuit with RZ angles fixed to 0
    complex_weights = torch.zeros(n_layers * n_units * 2, dtype=torch.float64)
    complex_weights_view = complex_weights.reshape(n_layers, n_units, 2)
    complex_weights_view[:, :, 0] = torch.tensor(ry_angles.reshape(n_layers, n_units))
    x_complex = x_t.to(torch.complex128)  # real input, but complex dtype for the general circuit
    state_complex = tms.simulate_final_state(x_complex.real, complex_weights, n_units, n_layers, pairs)
    xyz_complex, _ = tms.xyz_features_from_state(state_complex, n_units)
    # xyz_complex columns are [X, Y, Z]; compare only X and Z against feats_real's [X, Z]
    x_from_complex = xyz_complex[:, :n_units]
    z_from_complex = xyz_complex[:, 2 * n_units:3 * n_units]
    feats_from_complex_xz = torch.cat([x_from_complex, z_from_complex], dim=1)

    err = (feats_real - feats_from_complex_xz).abs().max().item()
    assert err < 1e-8, f"RotationCascadeNet does not match general circuit at RZ=0: max|diff|={err}"
    print(f"  [check] RotationCascadeNet vs general complex circuit (RZ=0, real input): max|diff|={err:.2e} (expect <1e-8)")

    # PermutationCascadeNet: verify it genuinely does NOT preserve norm internally
    # (the whole point of this architecture), and that its bounded readout still
    # produces finite, sane values -- both checked directly, not assumed.
    torch.manual_seed(2)
    x_test = torch.randn(5, 2 ** n_units, dtype=torch.float64)
    perm_net = PermutationCascadeNet(n_units, n_layers, pairs)
    with torch.no_grad():
        raw_state = simulate_permutation_cascade(x_test, perm_net.weights, n_units, n_layers, pairs)
        raw_norms = raw_state.reshape(5, -1).norm(dim=-1)
    norm_spread = (raw_norms.max() - raw_norms.min()).item()
    assert norm_spread > 1e-3, ("PermutationCascadeNet's internal state norm is suspiciously constant -- "
                                 "the no-norm-preservation manipulation may not be taking effect")
    print(f"  [check] PermutationCascadeNet internal state norms vary across samples "
          f"(range={raw_norms.min().item():.3f}-{raw_norms.max().item():.3f}, spread={norm_spread:.3f}) "
          f"-- confirms norm is genuinely NOT preserved internally (unlike RotationCascadeNet)")
    with torch.no_grad():
        feats_perm = perm_net(x_test)
    assert torch.isfinite(feats_perm).all(), "PermutationCascadeNet produced non-finite features"
    assert feats_perm.abs().max().item() <= 1.0 + 1e-4, "PermutationCascadeNet readout not properly bounded"
    print(f"  [check] PermutationCascadeNet readout: finite, |value|<=1 (max={feats_perm.abs().max().item():.4f})")

    print("Self-test PASSED.\n")


if __name__ == "__main__":
    self_test()
