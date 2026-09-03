"""Experiment 60 (README Follow-up 88; Phase-2 peer-review roadmap,
responding to a 2-way convergent finding from the EIC and Devil's
Advocate reviewers): MNIST is R_manifold=1 exactly (Section 8.1) -- the
model-restricted mechanism this paper studies contributes nothing
there -- yet Section 8.5 previously called MNIST's static-vs-adaptive
result an "independent replication" of that same mechanism. This is a
genuine gap: the paper has never run the full static-vs-adaptive
pilot-recalibration pipeline on a system that is BOTH independent of
BloodMNIST AND actually in the restricted regime.

The VQE checkpoints (Experiment 24) already establish that n_a=4 is
severely restricted there (median R_manifold=0.071 across 10 seeds,
6 qubits / 72 params / ambient dim 255) -- more restricted than
BloodMNIST's own 0.328. This experiment closes the gap directly: apply
the SAME primary task-invariant mechanism (full-magnitude ring-ZZ) used
throughout the paper to the VQE "cluster" (converged) checkpoints, and
run the SAME static-calibrated-vs-pilot-adaptive completion pipeline
Table II and Section 6.2 use for BloodMNIST, at the same headline
10^5-shot pilot budget.

Usage:
    python run_experiment60_vqe_adaptive_recalibration.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.restricted_observability import compute_R_manifold, hermitian_log, pauli_coefficients
from manifold.mvp133_vqe_checkpoints import load_theta, rho_A_and_jacobian, N_QUBITS, N_LAYERS, _mvp
from measurement_simulator.pauli_pilot import simulate_pilot
from selector.reduced_state_plugin import select as route_a_select
from seeds import derive_seed
from run_experiment35_multiple_perturbation_mechanisms import apply_zz_scaled

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_A = 4
SEEDS = list(range(10))
STAGE = "cluster"
N_PILOT = 100_000
N_TEST_REPEATS = 20
EXPERIMENT_ID = "paper13_exp60_vqe_adaptive_recalibration_v1"


def _rho_after(theta_perturbed_flat, n_a):
    n_qubits = N_QUBITS
    dim_a, dim_b = 2 ** n_a, 2 ** (n_qubits - n_a)
    M = theta_perturbed_flat.reshape(dim_a, dim_b)
    return (M @ M.conj().T).numpy()


def run(smoke: bool):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pool = build_pool(N_A)
    seeds = SEEDS[:2] if smoke else SEEDS
    n_test = 4 if smoke else N_TEST_REPEATS

    rows = []
    for seed in seeds:
        ts = load_theta(seed, STAGE)
        rho_before, jac_before = rho_A_and_jacobian(ts, N_A)
        exact_score_before_offdiag = compute_R_manifold(rho_before, jac_before, pool).exact_score[pool.offdiag_idx]
        j_before_local = int(np.argmax(exact_score_before_offdiag))
        j_before_global = pool.offdiag_idx[j_before_local]

        ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
        rng_phi = np.random.RandomState(seed)
        phis = list(rng_phi.uniform(-1.5, 1.5, size=N_QUBITS))

        with torch.no_grad():
            state_before = _mvp.simulate_rep_state(ts.theta.detach(), N_QUBITS, N_LAYERS)
            flat_before = state_before.reshape(2 ** N_QUBITS)
            flat_after = apply_zz_scaled(flat_before, N_QUBITS, phis, ring_pairs, scale=1.0)
        rho_after_np = _rho_after(flat_after, N_A)

        # jac_after: true post-perturbation tangent geometry, oracle/evaluation only,
        # never used for selection -- matches this project's established convention.
        def rho_A_realstack(w):
            state = _mvp.simulate_rep_state(w, N_QUBITS, N_LAYERS)
            flat = state.reshape(2 ** N_QUBITS)
            flat2 = apply_zz_scaled(flat, N_QUBITS, phis, ring_pairs, scale=1.0)
            dim_a, dim_b = 2 ** N_A, 2 ** (N_QUBITS - N_A)
            M = flat2.reshape(dim_a, dim_b)
            rho = M @ M.conj().T
            return torch.stack([rho.real, rho.imag])

        theta_leaf = ts.theta.detach().clone().requires_grad_(True)
        jac_raw = torch.autograd.functional.jacobian(rho_A_realstack, theta_leaf)
        jac_after = (jac_raw[0] + 1j * jac_raw[1]).numpy()

        baseline_after = compute_R_manifold(rho_after_np, jac_after, pool).gamma_D_C
        static_after = compute_R_manifold(rho_after_np, jac_after, pool,
                                           extra_measured_idx=[j_before_global]).gamma_D_C
        exact_score_after_true = compute_R_manifold(rho_after_np, jac_after, pool).exact_score[pool.offdiag_idx]
        j_after_local = int(np.argmax(exact_score_after_true))
        j_after_global = pool.offdiag_idx[j_after_local]
        oracle_after = compute_R_manifold(rho_after_np, jac_after, pool,
                                           extra_measured_idx=[j_after_global]).gamma_D_C
        oracle_reduction = baseline_after - oracle_after
        static_recovery = (baseline_after - static_after) / oracle_reduction if oracle_reduction > 1e-9 else float("nan")

        adaptive_recoveries = []
        for rep in range(n_test):
            s = derive_seed(EXPERIMENT_ID, f"vqe_seed{seed}", f"pilot_{rep}", "pilot")
            rng = np.random.default_rng(s)
            pilot = simulate_pilot(rho_after_np, pool.all_matrices, N_PILOT, rng)
            route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
            g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)
            exact_score_hat_offdiag = compute_R_manifold(
                route_a_out.rho_hat, jac_before, pool, g_coef_override=g_hat_coef
            ).exact_score[pool.offdiag_idx]
            j_adapt_local = int(np.argmax(exact_score_hat_offdiag))
            j_adapt_global = pool.offdiag_idx[j_adapt_local]
            adaptive_after = compute_R_manifold(rho_after_np, jac_after, pool,
                                                 extra_measured_idx=[j_adapt_global]).gamma_D_C
            recovery = (baseline_after - adaptive_after) / oracle_reduction if oracle_reduction > 1e-9 else float("nan")
            adaptive_recoveries.append(recovery)

        adaptive_recovery_mean = float(np.mean(adaptive_recoveries))
        row = dict(seed=seed, R_manifold_before=compute_R_manifold(rho_before, jac_before, pool).R_manifold,
                   baseline_after=baseline_after, oracle_reduction=oracle_reduction,
                   static_recovery=static_recovery, adaptive_recovery_mean=adaptive_recovery_mean,
                   j_flipped=(j_before_global != j_after_global))
        rows.append(row)
        print(f"seed={seed}: R_manifold_before={row['R_manifold_before']:.4f} "
              f"static={static_recovery*100:.2f}% adaptive={adaptive_recovery_mean*100:.2f}% "
              f"j_flipped={row['j_flipped']}", flush=True)
        _write_csv(RESULTS_DIR / "vqe_adaptive_recalibration.csv", rows)

    print(f"\nDone. {len(rows)} rows written.")
    static_mean = np.mean([r["static_recovery"] for r in rows])
    adaptive_mean = np.mean([r["adaptive_recovery_mean"] for r in rows])
    print(f"\n=== VQE (n_a={N_A}, restricted regime) summary, n={len(rows)} seeds ===")
    print(f"  mean static recovery:   {static_mean*100:.2f}%")
    print(f"  mean adaptive recovery: {adaptive_mean*100:.2f}%")
    print(f"  per-seed static:  {[round(r['static_recovery']*100,2) for r in rows]}")
    print(f"  per-seed adaptive: {[round(r['adaptive_recovery_mean']*100,2) for r in rows]}")


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
