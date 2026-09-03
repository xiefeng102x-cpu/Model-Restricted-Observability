"""Reconstructs the trained circuit parameters theta behind this
project's 20 persisted QML reduced states (states/qml_reduced_states.py),
and computes d(rho_A)/d(theta) -- the tangent-space Jacobian the
manifold-restricted observability direction needs.

theta is NOT persisted anywhere: the original state-generation script
only saves rho_before/rho_after (see that script's `np.savez` call).
Confirmed via direct inspection that no checkpoint file exists for
these specific 10 models.

theta reconstruction is deterministic and cheap to verify: `di.make_model`
calls torch.manual_seed(seed) before drawing theta's random init, and
`c1.phase1_pretrain_generic` trains it with no other randomness sources
than that same seed, so re-running the exact recipe reproduces the exact
trained theta -- checked at the call site (run_experiment23) by comparing
the reproduced final_ca against the saved .npz's own final_ca field, not
assumed.

"before"/"after" share the SAME theta: "after" is a deterministic,
seed-derived diagonal ZZ-null-unitary applied to the OUTPUT statevector,
not a retrained/fine-tuned model (mnist_null_intervention.apply_zz_null_
unitary; phis = list(np.random.RandomState(seed).uniform(-1.5, 1.5,
size=len(ring_pairs)))) -- this file reproduces that exact phis draw for
"after" so the reconstructed state matches the saved rho_after.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

tms = importlib.import_module("topology_mechanism_sweep")
c1 = importlib.import_module("candidate1_architecture_ladder_mnist")
mni = importlib.import_module("mnist_null_intervention")
di = importlib.import_module("diagnostic_identifiability")


@dataclass
class ThetaState:
    dataset: str
    seed: int
    theta: torch.Tensor           # (n_params,) float64, detached leaf, requires_grad
    n_qubits: int
    layer: int
    ring_pairs: list
    n_a: int
    x_ref: torch.Tensor           # (1, feat_dim) float64
    final_ca: float
    phis: list                    # ZZ-null-unitary angles used for the "after" stage


def reconstruct_theta(dataset_name: str, seed: int) -> ThetaState:
    cfg = di.DATASETS[dataset_name]
    x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
    model = di.make_model(dataset_name, seed)
    head, hist = c1.phase1_pretrain_generic(model, x_ct, x_cnt, seed, dataset_name)
    final_ca = float(hist[-1]["ca"])

    n_qubits, layer, ring_pairs = model.n_qubits, model.layer, model.ring_pairs
    x_ref = x_ct[:1]
    n_a = n_qubits // 2

    rng_phi = np.random.RandomState(seed)
    phis = list(rng_phi.uniform(-1.5, 1.5, size=len(ring_pairs)))

    theta = model.weights.detach().clone().requires_grad_(True)
    return ThetaState(dataset=dataset_name, seed=seed, theta=theta, n_qubits=n_qubits,
                       layer=layer, ring_pairs=ring_pairs, n_a=n_a, x_ref=x_ref,
                       final_ca=final_ca, phis=phis)


def rho_A_and_jacobian(ts: ThetaState, stage: str, intervention_fn=None):
    """stage in {'before','after'}. Returns (rho_A, jac):
    rho_A: (d,d) complex128 numpy, d = 2**ts.n_a.
    jac:   (d,d,n_params) complex128 numpy, jac[:,:,i] = d(rho_A)/d(theta_i)
           (Hermitian for every i, since rho_A is Hermitian-valued and
           theta is real).

    intervention_fn: optional callable (flat_state: (dim,) complex128
    tensor) -> (dim,) complex128 tensor, applied instead of the default
    ZZ-null-unitary when stage=='after' -- lets Follow-up 46 test OTHER
    task-invariant (diagonal-unitary) mechanisms through the exact same
    Jacobian machinery, without duplicating this function. Must be
    autograd-safe (pure elementwise ops on the state, no data-dependent
    control flow) since it sits inside the differentiated closure."""
    assert stage in ("before", "after")
    n_qubits, layer, ring_pairs = ts.n_qubits, ts.layer, ts.ring_pairs
    n_a = ts.n_a
    dim_a, dim_b = 2 ** n_a, 2 ** (n_qubits - n_a)
    x_ref, phis = ts.x_ref, ts.phis

    def rho_A_realstack(w):
        state = tms.simulate_final_state(x_ref, w, n_qubits, layer, ring_pairs)
        flat = state.reshape(2 ** n_qubits)
        if stage == "after":
            if intervention_fn is not None:
                flat = intervention_fn(flat)
            else:
                flat = mni.apply_zz_null_unitary(flat.unsqueeze(0), n_qubits, phis, ring_pairs).reshape(-1)
        M = flat.reshape(dim_a, dim_b)
        rho = M @ M.conj().T
        return torch.stack([rho.real, rho.imag])   # (2, dim_a, dim_a) real dtype, needed by jacobian()

    theta = ts.theta.detach().clone().requires_grad_(True)
    with torch.no_grad():
        rs0 = rho_A_realstack(theta)
    rho_A = (rs0[0] + 1j * rs0[1]).numpy()

    jac = torch.autograd.functional.jacobian(rho_A_realstack, theta)   # (2,dim_a,dim_a,n_params)
    jac_complex = (jac[0] + 1j * jac[1]).numpy()
    return rho_A, jac_complex


def self_test():
    print("Running self-test (manifold/theta_jacobian)...")
    torch.manual_seed(0)
    n_qubits, layer = 3, 2
    ring_pairs = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    weights = torch.randn(layer * n_qubits * 2, dtype=torch.float64) * 0.3
    x_ref = torch.randn(1, 8, dtype=torch.float64)
    n_a = 1
    dim_a, dim_b = 2 ** n_a, 2 ** (n_qubits - n_a)

    def rho_A_realstack(w):
        state = tms.simulate_final_state(x_ref, w, n_qubits, layer, ring_pairs)
        flat = state.reshape(2 ** n_qubits)
        M = flat.reshape(dim_a, dim_b)
        rho = M @ M.conj().T
        return torch.stack([rho.real, rho.imag])

    theta = weights.clone().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(rho_A_realstack, theta)

    # Hermiticity check: every d(rho_A)/d(theta_i) slice must itself be
    # Hermitian, since rho_A(theta) is Hermitian-valued for every real theta.
    jac_c = (jac[0] + 1j * jac[1]).detach().numpy()
    herm_err = np.abs(jac_c - jac_c.conj().transpose(1, 0, 2)).max()
    assert herm_err < 1e-10, f"d(rho_A)/d(theta_i) should be Hermitian, max asymmetry {herm_err}"
    print(f"  [check 1] every d(rho_A)/d(theta_i) slice is Hermitian (max asym={herm_err:.2e})")

    # Finite-difference cross-check against autograd, a few parameter indices.
    eps = 1e-6
    max_err = 0.0
    with torch.no_grad():
        for p_idx in [0, 5, 10, weights.numel() - 1]:
            wp = theta.detach().clone()
            wp[p_idx] += eps
            plus = rho_A_realstack(wp).numpy()
            wm = theta.detach().clone()
            wm[p_idx] -= eps
            minus = rho_A_realstack(wm).numpy()
            fd = (plus - minus) / (2 * eps)
            ad = jac[..., p_idx].numpy()
            err = np.abs(fd - ad).max()
            max_err = max(max_err, err)
    assert max_err < 1e-5, f"autograd Jacobian vs finite-difference mismatch: {max_err}"
    print(f"  [check 2] autograd d(rho_A)/d(theta) matches finite-difference (max err={max_err:.2e})")
    print("Self-test PASSED.\n")


if __name__ == "__main__":
    self_test()
