"""Experiment 36 (README Follow-up 47): Follow-up 46's own flagged next
step -- only two magnitude points (10% and 100% of the original ZZ-null-
unitary's phis) were compared; where does the actual transition between
"static calibration still works" and "static calibration fails" occur?
A systematic magnitude sweep, same ring-topology two-body ZZ generator
throughout (isolating magnitude as the only varying factor, as Follow-up
46 established it's the key driver, not entanglement structure).

Reuses Experiment 35's `apply_zz_scaled` and the `intervention_fn`
mechanism (manifold/theta_jacobian.py) unchanged; all 5 real BloodMNIST
seeds; each scale level's task-invariance follows automatically from
Follow-up 46's own already-verified weak-ZZ self-test (same construction,
different scale factor -- diagonal-in-computational-basis for any scale).

Usage:
    python run_experiment36_magnitude_sweep.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.theta_jacobian import ThetaState, rho_A_and_jacobian, di
from manifold.restricted_observability import compute_R_manifold, hermitian_log, pauli_coefficients
from measurement_simulator.pauli_pilot import simulate_pilot
from selector.reduced_state_plugin import select as route_a_select
from seeds import derive_seed
from run_experiment35_multiple_perturbation_mechanisms import apply_zz_scaled

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
N_QUBITS = 8
SEEDS = [42, 43, 44, 45, 46]
SCALES = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
ADAPTIVE_N_PILOT = 100_000
ADAPTIVE_N_REPEATS = 10
EXPERIMENT_ID = "paper13_exp36_magnitude_sweep_v1"


def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def run(smoke: bool):
    pool = build_pool(N_A)
    out_path = RESULTS_DIR / "magnitude_sweep.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    scales = [0.03, 1.0] if smoke else SCALES
    n_repeats = 3 if smoke else ADAPTIVE_N_REPEATS

    rows = []
    for scale in scales:
        for seed in seeds:
            art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
            theta0 = torch.tensor(art_before["theta"], dtype=torch.float64, requires_grad=True)
            rho_before, jac_before = art_before["rho_A"], art_before["jac"]

            cfg = di.DATASETS[DATASET]
            x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
            ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
            rng_phi = np.random.RandomState(seed)
            phis = list(rng_phi.uniform(-1.5, 1.5, size=N_QUBITS))

            ts = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=N_QUBITS, layer=8,
                             ring_pairs=ring_pairs, n_a=N_A, x_ref=x_ct[:1], final_ca=float("nan"), phis=phis)
            intervention_fn = lambda flat, phis=phis, ring_pairs=ring_pairs, scale=scale: apply_zz_scaled(
                flat, N_QUBITS, phis, ring_pairs, scale)
            rho_after, jac_after = rho_A_and_jacobian(ts, "after", intervention_fn=intervention_fn)

            res_before = compute_R_manifold(rho_before, jac_before, pool)
            res_after = compute_R_manifold(rho_after, jac_after, pool)
            cos_sim = _cos(res_before.r_DC_coef, res_after.r_DC_coef)

            j_before_local = int(np.argmax(res_before.exact_score[pool.offdiag_idx]))
            j_after_local = int(np.argmax(res_after.exact_score[pool.offdiag_idx]))
            same_j = j_before_local == j_after_local

            baseline_after = res_after.gamma_D_C
            j_before_global = pool.offdiag_idx[j_before_local]
            clean_calibrated = compute_R_manifold(rho_after, jac_after, pool,
                                                    extra_measured_idx=[j_before_global]).gamma_D_C
            j_after_global = pool.offdiag_idx[j_after_local]
            oracle_after = compute_R_manifold(rho_after, jac_after, pool,
                                               extra_measured_idx=[j_after_global]).gamma_D_C
            oracle_reduction = baseline_after - oracle_after
            clean_reduction = baseline_after - clean_calibrated
            clean_recovery = clean_reduction / oracle_reduction if oracle_reduction > 1e-9 else float("nan")

            adaptive_vals = []
            for rep in range(n_repeats):
                pilot_seed = derive_seed(EXPERIMENT_ID, f"scale{scale}_{DATASET}_seed{seed}",
                                          f"{ADAPTIVE_N_PILOT}_{rep}", "pilot")
                rng = np.random.default_rng(pilot_seed)
                pilot = simulate_pilot(rho_after, pool.all_matrices, ADAPTIVE_N_PILOT, rng)
                route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
                g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)
                adaptive_res = compute_R_manifold(rho_after, jac_before, pool, g_coef_override=g_hat_coef)
                j_adapt_local = int(np.argmax(adaptive_res.exact_score[pool.offdiag_idx]))
                j_adapt_global = pool.offdiag_idx[j_adapt_local]
                eval_res = compute_R_manifold(rho_after, jac_after, pool, extra_measured_idx=[j_adapt_global])
                adaptive_vals.append(eval_res.gamma_D_C)
            adaptive_mean = float(np.mean(adaptive_vals))
            adaptive_reduction = baseline_after - adaptive_mean
            adaptive_recovery = adaptive_reduction / oracle_reduction if oracle_reduction > 1e-9 else float("nan")

            row = dict(scale=scale, seed=seed, same_j_star=same_j, cos_r_DC=cos_sim,
                       clean_recovery_fraction=clean_recovery, adaptive_recovery_fraction=adaptive_recovery)
            rows.append(row)
            print(f"scale={scale:>5} seed={seed}: same_j*={same_j} cos={cos_sim:.4f} "
                  f"clean_recovery={clean_recovery*100:.1f}% adaptive_recovery={adaptive_recovery*100:.1f}%",
                  flush=True)
        _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}.")
    _summarize(rows, scales)


def _summarize(rows, scales):
    print("\n=== Magnitude sweep summary (mean across 5 seeds per scale) ===")
    print(f"{'scale':>7} {'|cos| mean':>12} {'j*same':>8} {'clean_recovery':>16} {'adaptive_recovery':>18}")
    for scale in scales:
        sub = [r for r in rows if r["scale"] == scale]
        if not sub:
            continue
        cos_vals = [abs(r["cos_r_DC"]) for r in sub]
        n_same = sum(1 for r in sub if r["same_j_star"])
        clean_vals = [r["clean_recovery_fraction"] for r in sub]
        adap_vals = [r["adaptive_recovery_fraction"] for r in sub]
        print(f"{scale:>7} {np.mean(cos_vals):>12.4f} {f'{n_same}/{len(sub)}':>8} "
              f"{np.mean(clean_vals)*100:>15.1f}% {np.mean(adap_vals)*100:>17.1f}%")


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
