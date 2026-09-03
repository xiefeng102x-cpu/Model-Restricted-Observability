"""
MVP-133 Phase 3: entanglement-capacity diagnostic under REAL shot
noise, closing a gap acknowledged in mvp133_entanglement_capacity_diagnostic.py.

That script's Case B (n=12, L=30, wide init -- an attempted genuine Cerezo/
McClean barren plateau) used EXACT, noiseless gradients. Result: Adam still
made smooth progress, and the entanglement-entropy trajectory did not show
the flat-from-the-start signature the barren-plateau hypothesis predicted --
acknowledged at the time as "exact simulation doesn't replicate real
shot-noise-driven barren-plateau stagnation," an open methodological gap.

This script closes that gap using a tool already built and validated
elsewhere in this project: app_noise_triggered_backdoor_test.py's analytic
shot-noise model, Var[O_hat] = (1 - <O>^2) / N_shots for any +-1-eigenvalue
observable (validated there to 0.67% relative error against real multinomial
sampling). That derivation is observable-agnostic -- it holds for ANY
Pauli observable with eigenvalues +-1 (a standard binomial-estimator
variance result: for O_hat = (n_+ - n_-)/N with p_+ - p_- = <O>, Var(O_hat)
= 4 p_+(1-p_+)/N = (1-<O>^2)/N), so it is reused HERE unmodified for each
individual term of the TFIM Hamiltonian (n ZZ terms + n X terms, each with
eigenvalues +-1), not just a single-qubit Z as in the original application.

Design: train VQE on Case A (verified expressivity-limited, n=6/L=6/g=0.5)
and Case B (n=12/L=30/wide-init, attempted genuine barren plateau) using a
NOISY energy estimate -- each Hamiltonian term's expectation gets
independent analytic shot noise added (std detached/stop-gradient, matching
established practice: don't let the optimizer game the noise MAGNITUDE
itself, only follow the noisy MEAN estimate) -- instead of the exact energy,
across a shot-count sweep [None(exact, reproduces the original result as a
baseline check), 1000, 200, 50] to see a dose-response curve, not just one
arbitrary noise level.

Question: does the (S_state trajectory shape, U=S_state/S_max) diagnostic
FINALLY distinguish Case A's expressivity-ceiling signature (rises then
hard-plateaus, already confirmed) from a genuine noise-driven stagnation
signature in Case B (predicted: flat-low or erratic-noisy S_state, energy
fails to converge smoothly) once real shot noise is present -- something
the noiseless version could not show.

Standalone rebuild: VQE ansatz, entanglement entropy, and Case A/B
configuration match mvp133_entanglement_capacity_diagnostic.py's validated
formulas exactly (re-derived fresh in this file, not imported). Shot-noise
injection matches app_noise_triggered_backdoor_test.py's validated formula
(re-derived fresh, not imported) generalized from a single Z-observable to
an arbitrary list of +-1-eigenvalue Pauli terms.

Self-tests:
  1. Entanglement entropy: product state -> 0, GHZ -> log(2), Haar-random
     state -> Page's-theorem value, differentiability (all four reused
     from the original diagnostic).
  2. Shot-noise formula: reuses the SAME analytic-vs-empirical-multinomial-
     sampling validation as app_noise_triggered_backdoor_test.py, but now
     applied to a MULTI-TERM Hamiltonian's per-term expectations (not just
     single-qubit Z) -- verifies the generalization is not silently wrong.
  3. Noisy-energy-at-N_shots=None reduces exactly to the noiseless energy
     (sanity check on the injection function's off-switch).

Outputs (results/):
  - entanglement_capacity_noisy_results.csv -- per (case, n_shots, seed,
    epoch): energy_noisy, energy_exact, S_state, U.
  - entanglement_capacity_noisy_summary.csv -- per (case, n_shots): S_init,
    S_final, U_final, rise_fraction, energy convergence quality.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS = 6  # self_test() only

N_QUBITS_A = 6
L_CASE_A = 6
G_CASE_A = 0.5
INIT_STD_A = 0.1

N_QUBITS_B = 12
L_CASE_B = 30
G_CASE_B = 0.5
INIT_WIDE_B = True

SEEDS = [0, 1, 2]
EXTRA_SEEDS = [3, 4]  # confirmation seeds, run via --seed CLI, merged in afterward
SMAX_PROBE_SEEDS = SEEDS + EXTRA_SEEDS  # fixed regardless of which training seeds run,
                                          # so S_max/U normalization stays comparable across separate runs
EPOCHS = 400
LR = 0.05
CHECKPOINTS = [0, 1, 2, 5, 10, 20, 40, 80, 120, 200, 300, 400]
S_MAX_TRAIN_STEPS = 300
SHOT_LEVELS = [None, 1000, 200, 50]  # None = exact/noiseless baseline (reproduces the original result)

def zz_term_eigenvalues(n: int, i: int, j: int):
    """Eigenvalues of Z_i Z_j across all 2**n computational basis states, as a
    length-2**n real vector -- ZZ is diagonal in the computational basis, so
    no (dim x dim) matrix is ever needed, just this eigenvalue vector."""
    dim = 2 ** n
    idx = np.arange(dim)
    bit_i = (idx >> (n - 1 - i)) & 1
    bit_j = (idx >> (n - 1 - j)) & 1
    return (1 - 2 * bit_i) * (1 - 2 * bit_j)


def tfim_term_spec(n: int, g: float):
    """Term-wise spec of the TFIM Hamiltonian (n ZZ terms + n X terms) that
    NEVER materializes a dense (2**n x 2**n) matrix per term. ZZ terms are
    diagonal -> represented as their (n, dim) eigenvalue matrix (coeff -1
    baked in). X terms are NOT diagonal but are single-qubit flips -> their
    expectation is computed at O(dim) via the same single-qubit-gate
    machinery the circuit simulator already uses, not a matrix-vector
    product against a dense matrix. This keeps memory O(n_terms * dim)
    instead of O(n_terms * dim^2): at n=12 (dim=4096) the old dense-per-term
    approach needed a (24, 4096, 4096) complex128 tensor (~6.4GB), which
    OOM'd on backward(); this representation needs ~400KB.
    Returns (zz_eigs: (n, dim) float64 ndarray, x_qubits: list[int], x_coeff: float)."""
    dim = 2 ** n
    zz_eigs = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        zz_eigs[i] = -1.0 * zz_term_eigenvalues(n, i, (i + 1) % n)
    x_qubits = list(range(n))
    x_coeff = -g
    return zz_eigs, x_qubits, x_coeff


def build_term_spec(n: int, g: float):
    zz_eigs, x_qubits, x_coeff = tfim_term_spec(n, g)
    zz_eigs_torch = torch.tensor(zz_eigs, dtype=torch.float64)
    return zz_eigs_torch, x_qubits, x_coeff


X_GATE = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)


def ry_matrix(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def rz_matrix(theta):
    e_m = torch.exp(-1j * theta / 2)
    e_p = torch.exp(1j * theta / 2)
    z = torch.zeros((), dtype=torch.complex128)
    return torch.stack([torch.stack([e_m, z]), torch.stack([z, e_p])])


def apply_single_qubit_gate(state, gate, qubit, n_qubits):
    axis = qubit
    state = torch.movedim(state, axis, 0)
    shape = state.shape
    state = state.reshape(2, -1)
    state = torch.einsum("ij,jk->ik", gate.to(state.dtype), state)
    state = state.reshape(shape)
    state = torch.movedim(state, 0, axis)
    return state


def apply_cnot(state, control, target, n_qubits):
    state = torch.movedim(state, (control, target), (0, 1))
    s0 = state[0].clone()
    s1 = state[1].clone()
    new_s1 = torch.flip(s1, dims=[0])
    state = torch.stack([s0, new_s1], dim=0)
    state = torch.movedim(state, (0, 1), (control, target))
    return state


def simulate_state(weights, n_qubits, n_layers):
    state = torch.zeros((2,) * n_qubits, dtype=torch.complex128)
    state.reshape(-1)[0] = 1.0
    wr = weights.reshape(n_layers, n_qubits, 2)
    for l in range(n_layers):
        for q in range(n_qubits):
            g_ry = ry_matrix(wr[l, q, 0].to(torch.float64)).to(torch.complex128)
            state = apply_single_qubit_gate(state, g_ry, q, n_qubits)
            g_rz = rz_matrix(wr[l, q, 1].to(torch.float64))
            state = apply_single_qubit_gate(state, g_rz, q, n_qubits)
        for q in range(n_qubits):
            state = apply_cnot(state, q, (q + 1) % n_qubits, n_qubits)
    return state.reshape(-1)


def von_neumann_entropy_bipartition(psi_flat, n_qubits, n_a, eps=1e-12):
    dim_a = 2 ** n_a
    dim_b = 2 ** (n_qubits - n_a)
    M = psi_flat.reshape(dim_a, dim_b)
    rho_a = M @ M.conj().T
    eigvals = torch.linalg.eigvalsh(rho_a).real
    eigvals = torch.clamp(eigvals, min=eps)
    entropy = -torch.sum(eigvals * torch.log(eigvals))
    return entropy


# ---------------------------------------------------------------- shot-noise-aware energy

def term_expectations(psi, term_spec, n_qubits):
    """Exact expectation value of every Hamiltonian term, computed without
    ever forming a (dim, dim) matrix. ZZ terms (diagonal): <ZZ> = probs @
    eigenvalues. X terms (single-qubit flip, O(dim)): <X_i> =
    Re(<psi|X_i|psi>) via the same apply_single_qubit_gate used by the
    circuit simulator."""
    zz_eigs_torch, x_qubits, x_coeff = term_spec
    probs = (psi.conj() * psi).real
    zz_vals = zz_eigs_torch @ probs
    state = psi.reshape((2,) * n_qubits)
    x_vals = []
    for q in x_qubits:
        psi_x = apply_single_qubit_gate(state, X_GATE, q, n_qubits).reshape(-1)
        x_vals.append(x_coeff * torch.real(torch.vdot(psi, psi_x)))
    return torch.cat([zz_vals, torch.stack(x_vals)])


def noisy_energy(psi, term_spec, n_qubits, n_shots, generator=None):
    """Returns noisy total energy = sum_i (exact <psi|h_i|psi> + noise_i), where
    noise_i ~ N(0, sqrt((1-<h_i>^2)/n_shots)), std DETACHED (stop-gradient on
    the noise magnitude, matching app_noise_triggered_backdoor_test.py's
    established practice). If n_shots is None, returns the exact energy.
    NOTE: this noisy VALUE is only for logging/diagnostics -- since noise is
    additive and independent of weights, adding it before backward() has
    IDENTICALLY ZERO effect on the gradient (d(noise)/d(weights)=0 always),
    so it must NOT be used as the training loss. See grad_noise_std() below
    for the function that actually injects shot noise into training."""
    term_expvals = term_expectations(psi, term_spec, n_qubits)
    if n_shots is None:
        return term_expvals.sum(), term_expvals
    var = (1.0 - term_expvals.detach() ** 2).clamp(min=0.0) / n_shots
    std = var.sqrt()
    noise = torch.randn(term_expvals.shape, generator=generator, dtype=term_expvals.dtype) * std
    noisy_terms = term_expvals + noise
    return noisy_terms.sum(), term_expvals


def grad_noise_std(term_expvals, n_shots):
    """Approximate isotropic per-parameter gradient-noise std induced by
    finite-shot measurement, so the noise actually reaches the optimizer.
    Derivation: on real hardware each gradient component is a finite
    difference of two shot-noisy measurements of the same term (parameter-
    shift rule), each with the SAME validated variance var_i=(1-<h_i>^2)/
    n_shots used for noisy_energy -- so Var(d term_i/d theta_k) ~= var_i/2
    as an order-of-magnitude proxy (uniform across params/terms; NOT exact
    per-parameter parameter-shift, which would need 2 extra circuit evals
    per parameter per term and is infeasible at Case B's ~720 params x 400
    epochs). Terms are measured independently on real hardware, so their
    variances add: Var(grad_total) ~= sum_i var_i / 2. Returns 0.0 for
    n_shots=None (exact/noiseless gradient, matching the pre-fix baseline)."""
    if n_shots is None:
        return 0.0
    var = (1.0 - term_expvals.detach() ** 2).clamp(min=0.0) / n_shots
    return (var.sum() / 2.0).sqrt().item()


def self_test():
    print("Running self-test (entanglement capacity with shot noise)...")
    n = N_QUBITS

    # 1. Entanglement entropy checks (reused from the original diagnostic)
    psi_prod = torch.zeros(2 ** n, dtype=torch.complex128); psi_prod[0] = 1.0
    s_prod = von_neumann_entropy_bipartition(psi_prod, n, n // 2)
    assert s_prod.item() < 1e-6
    print(f"  [check 1] product state entropy = {s_prod.item():.2e} (expect ~0)")

    psi_ghz = torch.zeros(2 ** n, dtype=torch.complex128)
    psi_ghz[0] = 1.0 / np.sqrt(2); psi_ghz[-1] = 1.0 / np.sqrt(2)
    s_ghz = von_neumann_entropy_bipartition(psi_ghz, n, n // 2)
    assert abs(s_ghz.item() - np.log(2)) < 1e-6
    print(f"  [check 2] GHZ state entropy = {s_ghz.item():.6f} (expect ln(2)={np.log(2):.6f})")

    torch.manual_seed(0)
    w = (torch.randn(6 * n * 2, dtype=torch.float64) * 0.3).requires_grad_(True)
    psi = simulate_state(w, n, 6)
    s = von_neumann_entropy_bipartition(psi, n, n // 2)
    grad = torch.autograd.grad(s, w)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum().item() > 0
    print(f"  [check 3] entropy differentiable: ||d(S)/d(theta)||={grad.norm().item():.4e}")

    # 2. Shot-noise formula validated against REAL empirical multinomial sampling,
    #    now for a MULTI-TERM Hamiltonian's per-term expectations (generalization check).
    term_spec4 = build_term_spec(4, 0.5)  # small n for a tractable exact multinomial check
    torch.manual_seed(1)
    w4 = torch.randn(3 * 4 * 2, dtype=torch.float64) * 0.5
    psi4 = simulate_state(w4, 4, 3)
    _, exact_terms = noisy_energy(psi4, term_spec4, 4, None)
    exact_terms_np = exact_terms.detach().numpy()

    probs = (psi4.conj() * psi4).real.numpy()
    probs = probs / probs.sum()
    n_shots_test, n_trials = 300, 20000
    rng = np.random.default_rng(0)
    samples = rng.choice(len(probs), size=(n_trials, n_shots_test), p=probs)
    # eigenvalues of each Pauli term on each basis state: diagonal of h_terms (all terms here are diagonal in Z or X basis... )
    # simplest robust check: term 0 is a ZZ term (diagonal in computational basis)
    dim = len(probs)
    state_indices = np.arange(dim)
    bits = ((state_indices[:, None] >> np.arange(4 - 1, -1, -1)) & 1)
    zz_eigs_term0 = (1 - 2 * bits[:, 0]) * (1 - 2 * bits[:, 1])  # Z_0 Z_1 eigenvalues -> term index 0 is Z_0Z_1 with coeff -1
    term0_eigs = -1.0 * zz_eigs_term0
    empirical_term0 = term0_eigs[samples].mean(axis=1)
    empirical_var = empirical_term0.var()
    analytic_var = (1.0 - exact_terms_np[0] ** 2) / n_shots_test
    rel_err = abs(empirical_var - analytic_var) / (analytic_var + 1e-8)
    assert rel_err < 0.15, f"shot-noise formula mismatch for multi-term Hamiltonian: rel_err={rel_err}"
    print(f"  [check 4] analytic vs empirical shot-noise variance (ZZ term of multi-term H): "
          f"rel err={rel_err:.4f} (n_trials={n_trials}, threshold 0.15)")

    # 3. n_shots=None reduces exactly to the noiseless energy.
    e_noisy_none, _ = noisy_energy(psi4, term_spec4, 4, None)
    e_exact = exact_terms_np.sum()
    assert abs(e_noisy_none.item() - e_exact) < 1e-10
    print(f"  [check 5] noisy_energy(n_shots=None) == exact energy: {e_noisy_none.item():.6f} vs {e_exact:.6f}")

    # 6. grad_noise_std must actually perturb the gradient that reaches the
    #    optimizer -- catches the bug where additive/detached noise on the
    #    LOSS VALUE has zero effect on backward() (d(noise)/d(weights)=0
    #    identically, since noise doesn't depend on weights at all).
    w4b = w4.clone().detach().requires_grad_(True)
    psi4b = simulate_state(w4b, 4, 3)
    term_expvals_b = term_expectations(psi4b, term_spec4, 4)
    energy_b = term_expvals_b.sum()
    w4b.grad = None
    energy_b.backward()
    grad_exact_only = w4b.grad.clone()
    gstd = grad_noise_std(term_expvals_b, 50)
    assert gstd > 0, "grad_noise_std should be > 0 for a finite n_shots"
    gen6 = torch.Generator().manual_seed(7)
    grad_perturbed = grad_exact_only + torch.randn(grad_exact_only.shape, generator=gen6, dtype=grad_exact_only.dtype) * gstd
    diff = (grad_perturbed - grad_exact_only).abs().max().item()
    assert diff > 1e-6, "gradient noise injection had no effect -- fix regressed"
    assert grad_noise_std(term_expvals_b, None) == 0.0
    print(f"  [check 6] grad_noise_std(n_shots=50)={gstd:.4e} > 0, actually perturbs grad "
          f"(max|diff|={diff:.4e}); grad_noise_std(n_shots=None)=0.0 (exact)")

    print("Self-test PASSED.\n")


def maximize_entanglement(n_qubits, n_layers, bipartition_a, seed, steps=S_MAX_TRAIN_STEPS):
    torch.manual_seed(seed + 5000)
    weights = (torch.randn(n_layers * n_qubits * 2, dtype=torch.float64) * 0.5).requires_grad_(True)
    opt = torch.optim.Adam([weights], lr=0.05)
    best = 0.0
    for step in range(steps):
        psi = simulate_state(weights, n_qubits, n_layers)
        s = von_neumann_entropy_bipartition(psi, n_qubits, bipartition_a)
        loss = -s
        opt.zero_grad(); loss.backward(); opt.step()
        best = max(best, s.item())
    return best


def train_and_track_noisy(seed, n_qubits, bipartition_a, n_layers, term_spec, init_wide, n_shots, epochs=EPOCHS,
                           init_weights=None, return_weights=False):
    """init_weights: optional pre-built (n_layers*n_qubits*2,) tensor to use INSTEAD of
    the random init (e.g. a shallower solution's weights concatenated with near-identity
    padding for extra layers, for warm-start experiments) -- when given, init_wide is
    ignored for weight construction (still used for RNG-seeding consistency elsewhere).
    return_weights: when True, returns (records, grad_norm_init, final_weights) instead
    of the original (records, grad_norm_init) -- opt-in so all existing callers that
    unpack a 2-tuple are unaffected."""
    torch.manual_seed(seed)
    if init_weights is not None:
        weights = init_weights.clone().detach().requires_grad_(True)
    elif init_wide:
        weights = (torch.rand(n_layers * n_qubits * 2, dtype=torch.float64) * 2 * np.pi).requires_grad_(True)
    else:
        weights = (torch.randn(n_layers * n_qubits * 2, dtype=torch.float64) * INIT_STD_A).requires_grad_(True)
    opt = torch.optim.Adam([weights], lr=LR)
    gen = torch.Generator().manual_seed(seed * 13 + (n_shots or 0))
    checkpoint_set = set(CHECKPOINTS)
    records = []
    grad_norm_init = None
    for epoch in range(epochs + 1):
        if epoch in checkpoint_set:
            with torch.no_grad():
                psi = simulate_state(weights, n_qubits, n_layers)
                _, exact_terms = noisy_energy(psi, term_spec, n_qubits, None)
                energy_exact = exact_terms.sum().item()
                s_state = von_neumann_entropy_bipartition(psi, n_qubits, bipartition_a).item()
            records.append({"seed": seed, "epoch": epoch, "energy_exact": energy_exact, "S_state": s_state})
        if epoch == epochs:
            break
        psi = simulate_state(weights, n_qubits, n_layers)
        term_expvals = term_expectations(psi, term_spec, n_qubits)
        energy_exact_train = term_expvals.sum()
        opt.zero_grad(); energy_exact_train.backward()
        gstd = grad_noise_std(term_expvals, n_shots)
        if gstd > 0:
            with torch.no_grad():
                weights.grad += torch.randn(weights.grad.shape, generator=gen, dtype=weights.grad.dtype) * gstd
        if epoch == 0:
            grad_norm_init = weights.grad.norm().item()
        opt.step()
    if return_weights:
        return records, grad_norm_init, weights.detach().clone()
    return records, grad_norm_init


def main(seed_filter=None):
    t_start = time.time()
    self_test()

    seeds_to_run = seed_filter if seed_filter is not None else SEEDS
    suffix = f"_seeds{'-'.join(str(s) for s in seeds_to_run)}" if seed_filter is not None else ""

    bip_A = N_QUBITS_A // 2
    bip_B = N_QUBITS_B // 2
    term_spec_A = build_term_spec(N_QUBITS_A, G_CASE_A)
    term_spec_B = build_term_spec(N_QUBITS_B, G_CASE_B)

    print("=== S_max capacity probes (fixed probe-seed set, independent of which training seeds run) ===")
    smax_A = np.mean([maximize_entanglement(N_QUBITS_A, L_CASE_A, bip_A, seed) for seed in SMAX_PROBE_SEEDS])
    smax_B = np.mean([maximize_entanglement(N_QUBITS_B, L_CASE_B, bip_B, seed) for seed in SMAX_PROBE_SEEDS])
    print(f"  Case A: S_max={smax_A:.4f}  Case B: S_max={smax_B:.4f}")

    all_rows = []
    cases = [
        ("A", N_QUBITS_A, bip_A, L_CASE_A, term_spec_A, False, smax_A),
        ("B", N_QUBITS_B, bip_B, L_CASE_B, term_spec_B, INIT_WIDE_B, smax_B),
    ]
    for case_name, n_q, bip, L, term_spec, wide, smax in cases:
        for n_shots in SHOT_LEVELS:
            print(f"\n=== Case {case_name}, n_shots={n_shots} ===")
            for seed in seeds_to_run:
                t0 = time.time()
                records, grad_init = train_and_track_noisy(seed, n_q, bip, L, term_spec, wide, n_shots)
                for r in records:
                    r.update({"case": case_name, "n_shots": "exact" if n_shots is None else n_shots,
                               "L": L, "S_max": smax, "U": r["S_state"] / smax if smax > 1e-9 else float("nan"),
                               "grad_norm_init": grad_init})
                    all_rows.append(r)
                print(f"  seed={seed} ({time.time()-t0:.1f}s): grad_init={grad_init:.4e}, "
                      f"final energy={records[-1]['energy_exact']:.4f}, final S_state={records[-1]['S_state']:.4f}, "
                      f"final U={records[-1]['S_state']/smax:.3f}")
            pd.DataFrame(all_rows).to_csv(OUT_DIR / f"entanglement_capacity_noisy_results{suffix}.csv", index=False)

    df = pd.DataFrame(all_rows)
    print(f"\n\n=== SUMMARY (total wall time {time.time()-t_start:.1f}s) ===")
    summary_rows = []
    for case_name in ["A", "B"]:
        for n_shots_label in df[df["case"] == case_name]["n_shots"].unique():
            sub = df[(df["case"] == case_name) & (df["n_shots"] == n_shots_label)]
            by_epoch = sub.groupby("epoch")[["energy_exact", "S_state", "U"]].mean()
            s_init, s_final = by_epoch["S_state"].iloc[0], by_epoch["S_state"].iloc[-1]
            u_final = by_epoch["U"].iloc[-1]
            rise_fraction = (s_final - s_init) / max(s_final, 1e-9)
            e_init, e_final = by_epoch["energy_exact"].iloc[0], by_epoch["energy_exact"].iloc[-1]
            summary_rows.append({"case": case_name, "n_shots": n_shots_label,
                                   "S_init": s_init, "S_final": s_final, "U_final": u_final,
                                   "rise_fraction": rise_fraction, "energy_init": e_init, "energy_final": e_final,
                                   "mean_grad_norm_init": sub.drop_duplicates("seed")["grad_norm_init"].mean()})
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUT_DIR / f"entanglement_capacity_noisy_summary{suffix}.csv", index=False)

    print("\n=== Does adding real shot noise reveal a stagnation signature in Case B that the noiseless version missed? ===")
    print("Compare rise_fraction and final energy convergence across n_shots within Case B:")
    print(summary_df[summary_df["case"] == "B"][["n_shots", "S_init", "S_final", "rise_fraction", "energy_final"]].to_string(index=False))


if __name__ == "__main__":
    import sys
    if "--self-test-only" in sys.argv:
        self_test()
    elif "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        seed_args = [int(s) for s in sys.argv[idx + 1:] if s.lstrip("-").isdigit()]
        main(seed_filter=seed_args)
    else:
        main()
