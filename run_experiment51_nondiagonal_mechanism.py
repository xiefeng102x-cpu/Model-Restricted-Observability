"""Experiment 51 (README Follow-up 63; manuscript Threat Model,
"Explicitly out of scope": "non-diagonal task-preserving
transformations"). Every perturbation tested elsewhere in this project
is a DIAGONAL unitary -- exactly task-invariant for EVERY state, not
just the specific deployed one. This is not an arbitrary restriction:
a standard Lie-theory fact (the centralizer of a maximal abelian
subalgebra in U(d) is the subalgebra itself) means diagonal unitaries
are close to the ONLY family that is EXACTLY task-invariant for ALL
states simultaneously, given this project's own measured set (the full
diagonal Pauli algebra on the audited subsystem). Testing a genuinely
non-diagonal mechanism therefore requires relaxing "invariant for every
state" to "invariant for the specific deployed state" -- arguably MORE
realistic anyway, since a real attacker only needs to evade detection
at the specific model instance they are attacking, not universally.

Construction: given the trusted reference state rho_before, find the
null space of the linear map H -> diag-Pauli-coefficients of i[H,rho]
restricted to OFF-DIAGONAL-Pauli-supported generators H (a (15, 240)
real matrix for n_a=4, generically with a large, >=225-dimensional null
space) -- any H drawn from this null space satisfies, EXACTLY (not
approximately), d/dt diag(rho(t))|_0 = 0 for U(t)=exp(-itH), i.e. FIRST-
ORDER task invariance at the reference state. The finite-t residual
drift (a real, nonzero, honestly-measured quantity, unlike the diagonal
family's exact-for-all-t invariance) is reported explicitly, not hidden.

Applied directly to rho_A (a "local-to-A" attack, physically the
complement of Follow-up 56/60's "local-to-B" trivial-exploit finding):
U = expm(-i*t*H), rho_after = U rho_before U^dagger. Reuses jac_before
as the tangent-geometry reference throughout (both for the oracle and
static-selection evaluation), consistent with every other "realistic"
pipeline in this project, since a true post-perturbation Jacobian is
not available in closed form for this state-level (not circuit-level)
construction.

Usage:
    python run_experiment51_nondiagonal_mechanism.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm, null_space

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.restricted_observability import compute_R_manifold, hermitian_log, pauli_coefficients
from measurement_simulator.pauli_pilot import simulate_pilot
from selector.reduced_state_plugin import select as route_a_select
from seeds import derive_seed

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
N_PILOT = 100_000
N_REPEATS = 20
ROTATION_SCALE = 1.0
EXPERIMENT_ID = "paper13_exp51_nondiagonal_v1"


def _commutator_diag_map(rho: np.ndarray, pool) -> np.ndarray:
    """C: (n_diag, n_offdiag) real, C @ h = diag-Pauli coefficients of
    i*[H,rho] for H = sum_j h_j * offdiag_matrices[j]."""
    diag_mats = pool.all_matrices[pool.diag_idx]
    n_diag, n_offdiag = len(pool.diag_idx), len(pool.offdiag_idx)
    C = np.zeros((n_diag, n_offdiag))
    for j, g_idx in enumerate(pool.offdiag_idx):
        Pj = pool.all_matrices[g_idx]
        comm = 1j * (Pj @ rho - rho @ Pj)
        C[:, j] = pauli_coefficients(comm, diag_mats)
    return C


def _build_nondiagonal_generator(rho_before, pool, rng):
    C = _commutator_diag_map(rho_before, pool)
    N = null_space(C, rcond=1e-8)
    assert N.shape[1] > 0, "expected a nontrivial null space of first-order-diagonal-preserving generators"
    coeffs = rng.normal(size=N.shape[1])
    h = N @ coeffs
    h = h / np.linalg.norm(h)
    H = np.zeros((pool.d, pool.d), dtype=complex)
    for j, g_idx in enumerate(pool.offdiag_idx):
        H += h[j] * pool.all_matrices[g_idx]
    herm_err = np.abs(H - H.conj().T).max()
    assert herm_err < 1e-10, f"generator should be exactly Hermitian, got max asymmetry {herm_err:.2e}"
    return H, N.shape[1]


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    n_repeats = 3 if smoke else N_REPEATS

    rows = []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        rho_before, jac_before = art_before["rho_A"], art_before["jac"]

        rng_gen = np.random.default_rng(derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", "generator", "mechanism"))
        H, null_dim = _build_nondiagonal_generator(rho_before, pool, rng_gen)

        U = expm(-1j * ROTATION_SCALE * H)
        unitary_err = np.abs(U @ U.conj().T - np.eye(pool.d)).max()
        rho_after = U @ rho_before @ U.conj().T

        diag_drift = float(np.abs(np.diag(rho_after).real - np.diag(rho_before).real).max())
        offdiag_change = float(np.abs(rho_after - rho_before).max())

        res_before = compute_R_manifold(rho_before, jac_before, pool)
        j_before = pool.offdiag_idx[int(np.argmax(res_before.exact_score[pool.offdiag_idx]))]

        baseline_after = compute_R_manifold(rho_after, jac_before, pool).gamma_D_C
        res_true_after = compute_R_manifold(rho_after, jac_before, pool)
        j_oracle = pool.offdiag_idx[int(np.argmax(res_true_after.exact_score[pool.offdiag_idx]))]
        j_flipped = j_oracle != j_before
        oracle_after = compute_R_manifold(rho_after, jac_before, pool, extra_measured_idx=[j_oracle]).gamma_D_C
        oracle_reduction = baseline_after - oracle_after
        static_after = compute_R_manifold(rho_after, jac_before, pool, extra_measured_idx=[j_before]).gamma_D_C
        static_recovery = (baseline_after - static_after) / oracle_reduction if oracle_reduction > 1e-9 else float("nan")

        adaptive_recoveries = []
        for rep in range(n_repeats):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"{rep}", "pilot")
            rng = np.random.default_rng(s)
            pilot = simulate_pilot(rho_after, pool.all_matrices, N_PILOT, rng)
            route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
            g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)
            sel_res = compute_R_manifold(rho_after, jac_before, pool, g_coef_override=g_hat_coef)
            j_hat = pool.offdiag_idx[int(np.argmax(sel_res.exact_score[pool.offdiag_idx]))]
            eval_after = compute_R_manifold(rho_after, jac_before, pool, extra_measured_idx=[j_hat]).gamma_D_C
            adaptive_recoveries.append((baseline_after - eval_after) / oracle_reduction
                                        if oracle_reduction > 1e-9 else float("nan"))

        row = dict(seed=seed, null_dim=null_dim, unitary_err=unitary_err,
                   diag_drift_maxabs=diag_drift, offdiag_change_maxabs=offdiag_change,
                   R_manifold_before=res_before.R_manifold, j_flipped=j_flipped,
                   static_recovery=static_recovery, adaptive_recovery=float(np.mean(adaptive_recoveries)))
        rows.append(row)
        print(f"seed={seed}: null_dim={null_dim} diag_drift={diag_drift:.2e} offdiag_change={offdiag_change:.4f} "
              f"R_manifold={res_before.R_manifold:.4f} flip={j_flipped} "
              f"static={static_recovery*100:5.1f}% adaptive={np.mean(adaptive_recoveries)*100:5.1f}%", flush=True)
        _write_csv(RESULTS_DIR / "nondiagonal_mechanism.csv", rows)

    print(f"\nDone. {len(rows)} rows written.")
    _summarize(rows)


def _summarize(rows):
    n_flip = sum(1 for r in rows if r["j_flipped"])
    print(f"\n=== Non-diagonal mechanism summary (mean across {len(rows)} seeds) ===")
    print(f"  max diag drift (should be small, NOT zero): {max(r['diag_drift_maxabs'] for r in rows):.2e}")
    print(f"  R_manifold={np.mean([r['R_manifold_before'] for r in rows]):.4f}")
    print(f"  j* flipped: {n_flip}/{len(rows)}")
    print(f"  static recovery={np.mean([r['static_recovery'] for r in rows])*100:.1f}%")
    print(f"  adaptive recovery={np.mean([r['adaptive_recovery'] for r in rows])*100:.1f}%")


def _write_csv(path: Path, rows: list):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(smoke=args.smoke)
