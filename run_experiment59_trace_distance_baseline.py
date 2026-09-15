"""Experiment 59 (README Follow-up 87; Phase-2 peer-review roadmap,
responding to a 3-way convergent finding from the Domain, Methodology,
and Devil's Advocate reviewers): Experiment 54's "naive baseline" (raw
off-diagonal Pauli-coefficient discrepancy) still passes the
pilot-reconstructed state through a nonlinear transform
($-\\log\\hat\\rho_A$) before scoring. Reviewers asked for a genuinely
naive baseline: does the pilot-reconstructed state simply *look
different* from the trusted reference at all, with no Jacobian, no
diagnostic gradient, no Pauli-coefficient decomposition -- just a
direct state-distance metric?

Design: reuses Experiment 38/54's exact protocol (dataset, 5 seeds,
6 conditions -- 5 ring-ZZ magnitudes + Z-only, 10^5-shot pilot,
calibration/test split). For every single pilot draw, computes THREE
scores from the SAME reconstructed state, so all three are paired on
identical pilot noise:
  (a) manifold-restricted r_DC discrepancy (Experiment 38's original
      score, requires jac_before);
  (b) naive off-diagonal Pauli-coefficient discrepancy (Experiment 54's
      score, no Jacobian, but still a -log(rho) transform);
  (c) trace distance between the pilot-reconstructed rho_hat and the
      trusted rho_before -- no Jacobian, no diagnostic, no Pauli
      decomposition at all, the most naive baseline possible.

Usage:
    python run_experiment59_trace_distance_baseline.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.theta_jacobian import ThetaState, rho_A_and_jacobian, di
from manifold.restricted_observability import compute_R_manifold, hermitian_log, pauli_coefficients
from measurement_simulator.pauli_pilot import simulate_pilot
from selector.reduced_state_plugin import select as route_a_select
from seeds import derive_seed
from run_experiment35_multiple_perturbation_mechanisms import apply_zz_scaled, apply_single_z_null_unitary

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
N_QUBITS = 8
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
ZZ_SCALES = [0.05, 0.15, 0.3, 0.6, 1.0]
N_PILOT = 100_000
N_CALIB = 15
N_TEST_CLEAN = 15
N_TEST_TAMPERED = 15
EXPERIMENT_ID = "paper13_exp59_trace_distance_baseline_v1"


def _trace_distance(rho_a, rho_b):
    diff = rho_a - rho_b
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(eigvals)))


def _three_scores(rho_true, rho_before, jac_before, r_dc_before_offdiag, g_before_offdiag, pool, n_pilot, rng):
    pilot = simulate_pilot(rho_true, pool.all_matrices, n_pilot, rng)
    route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
    rho_hat = route_a_out.rho_hat

    g_hat_coef = pauli_coefficients(-hermitian_log(rho_hat), pool.all_matrices)
    r_dc_hat_offdiag = compute_R_manifold(
        rho_hat, jac_before, pool, g_coef_override=g_hat_coef
    ).r_DC_coef[pool.offdiag_idx]
    score_manifold = float(np.linalg.norm(r_dc_hat_offdiag - r_dc_before_offdiag))

    g_hat_offdiag = g_hat_coef[pool.offdiag_idx]
    score_naive_offdiag = float(np.linalg.norm(g_hat_offdiag - g_before_offdiag))

    score_trace_distance = _trace_distance(rho_hat, rho_before)

    return score_manifold, score_naive_offdiag, score_trace_distance


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    zz_scales = [0.15, 1.0] if smoke else ZZ_SCALES
    n_calib = 4 if smoke else N_CALIB
    n_test_clean = 4 if smoke else N_TEST_CLEAN
    n_test_tampered = 4 if smoke else N_TEST_TAMPERED

    score_rows = []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        theta0 = torch.tensor(art_before["theta"], dtype=torch.float64, requires_grad=True)
        rho_before, jac_before = art_before["rho_A"], art_before["jac"]
        r_dc_before_offdiag = compute_R_manifold(rho_before, jac_before, pool).r_DC_coef[pool.offdiag_idx]
        g_before_coef = pauli_coefficients(-hermitian_log(rho_before), pool.all_matrices)
        g_before_offdiag = g_before_coef[pool.offdiag_idx]

        cfg = di.DATASETS[DATASET]
        x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
        ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
        rng_phi = np.random.RandomState(seed)
        phis = list(rng_phi.uniform(-1.5, 1.5, size=N_QUBITS))

        for rep in range(n_calib):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"calib_{rep}", "pilot")
            rng = np.random.default_rng(s)
            _three_scores(rho_before, rho_before, jac_before, r_dc_before_offdiag, g_before_offdiag, pool, N_PILOT, rng)

        for rep in range(n_test_clean):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"testclean_{rep}", "pilot")
            rng = np.random.default_rng(s)
            sm, sn, st = _three_scores(rho_before, rho_before, jac_before, r_dc_before_offdiag, g_before_offdiag, pool, N_PILOT, rng)
            score_rows.append(dict(seed=seed, condition="clean", label=0,
                                    score_manifold=sm, score_naive_offdiag=sn, score_trace_distance=st))
        print(f"seed={seed}: clean calibration ({n_calib}) + test ({n_test_clean}) done", flush=True)

        for scale in zz_scales:
            ts = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=N_QUBITS, layer=8,
                             ring_pairs=ring_pairs, n_a=N_A, x_ref=x_ct[:1], final_ca=float("nan"), phis=phis)
            intervention_fn = lambda flat, phis=phis, ring_pairs=ring_pairs, scale=scale: apply_zz_scaled(
                flat, N_QUBITS, phis, ring_pairs, scale)
            rho_after, jac_after = rho_A_and_jacobian(ts, "after", intervention_fn=intervention_fn)
            for rep in range(n_test_tampered):
                s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"zz{scale}_{rep}", "pilot")
                rng = np.random.default_rng(s)
                sm, sn, st = _three_scores(rho_after, rho_before, jac_before, r_dc_before_offdiag, g_before_offdiag, pool, N_PILOT, rng)
                score_rows.append(dict(seed=seed, condition=f"zz_scale_{scale}", label=1,
                                        score_manifold=sm, score_naive_offdiag=sn, score_trace_distance=st))
            print(f"  zz scale={scale}: {n_test_tampered} tampered draws done", flush=True)

        ts_z = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=N_QUBITS, layer=8,
                           ring_pairs=ring_pairs, n_a=N_A, x_ref=x_ct[:1], final_ca=float("nan"), phis=phis)
        z_thetas = list(np.random.RandomState(seed + 90001).uniform(-1.5, 1.5, size=N_QUBITS))
        z_fn = lambda flat, z_thetas=z_thetas: apply_single_z_null_unitary(flat, N_QUBITS, z_thetas)
        rho_z, jac_z = rho_A_and_jacobian(ts_z, "after", intervention_fn=z_fn)
        for rep in range(n_test_tampered):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"zonly_{rep}", "pilot")
            rng = np.random.default_rng(s)
            sm, sn, st = _three_scores(rho_z, rho_before, jac_before, r_dc_before_offdiag, g_before_offdiag, pool, N_PILOT, rng)
            score_rows.append(dict(seed=seed, condition="z_only", label=1,
                                    score_manifold=sm, score_naive_offdiag=sn, score_trace_distance=st))
        print(f"  z_only: {n_test_tampered} tampered draws done", flush=True)

        _write_csv(RESULTS_DIR / "trace_distance_baseline_scores.csv", score_rows)

    print(f"\nDone collecting {len(score_rows)} scored trials.")
    _summarize(score_rows)


def _summarize(rows):
    labels_all = np.array([r["label"] for r in rows])
    for key, name in [("score_manifold", "manifold-restricted (r_DC discrepancy)"),
                       ("score_naive_offdiag", "naive off-diagonal Pauli discrepancy"),
                       ("score_trace_distance", "trace distance (no Jacobian, no diagnostic)")]:
        scores = np.array([r[key] for r in rows])
        auc = roc_auc_score(labels_all, scores)
        fpr, tpr, _ = roc_curve(labels_all, scores)

        def tpr_at_fpr(target_fpr, fpr=fpr, tpr=tpr):
            idx = max(np.searchsorted(fpr, target_fpr, side="right") - 1, 0)
            return float(tpr[idx])

        print(f"\n=== Pooled end-to-end detection -- {name} ===")
        print(f"  ROC-AUC = {auc:.4f}")
        print(f"  TPR@1%FPR = {tpr_at_fpr(0.01)*100:.1f}%  TPR@5%FPR = {tpr_at_fpr(0.05)*100:.1f}%")

        clean_scores = scores[labels_all == 0]
        print(f"  Per-condition AUC:")
        for cond in sorted(set(r["condition"] for r in rows if r["label"] == 1)):
            cond_scores = np.array([r[key] for r in rows if r["condition"] == cond])
            y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(cond_scores))])
            s = np.concatenate([clean_scores, cond_scores])
            try:
                cond_auc = roc_auc_score(y, s)
            except ValueError:
                cond_auc = float("nan")
            print(f"    {cond:>16}: AUC={cond_auc:.4f}")


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
