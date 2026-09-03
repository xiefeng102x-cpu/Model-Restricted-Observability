"""Experiment 54 (README Follow-up 79; Phase-2 peer-review roadmap item
P1-1, responding to the Devil's Advocate's C4 and the Domain reviewer's
independently-raised "application depth" question): does the manifold-
restricted machinery (Theorem 1's tangent-space projection, which
requires the model's own Jacobian) actually earn its keep in the
end-to-end detector Experiment 38 reports, or would a naive score built
from nothing but the raw, un-projected off-diagonal Pauli-coefficient
discrepancy -- no jac, no tangent-space projection, no null-space
computation at all -- do just as well?

Design: reuses Experiment 38's exact protocol (dataset, seeds,
ring-ZZ magnitude sweep, Z-only mechanism, pilot budget, calibration/
test split) so the two scores are compared under an identical
measurement budget. For every single pilot draw, BOTH scores are
computed from the SAME simulated/reconstructed state -- a paired
comparison, not independently redrawn -- so any AUC difference reflects
the choice of statistic, not incidental pilot noise:

  - existing score (Experiment 37/38's r_DC discrepancy): requires
    jac_before; projects the diagnostic gradient onto the model's own
    reachable tangent space intersected with the measured-operator null
    space (guide Theorem B's r_{D,C} construction).
  - naive baseline score: ||g_hat_offdiag - g_before_offdiag||, the
    pilot-reconstructed state's raw ambient off-diagonal Pauli
    coefficients compared directly to the trusted clean baseline's --
    no jac needed anywhere.

Usage:
    python run_experiment54_naive_diagnostic_baseline.py [--smoke]
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
SEEDS = [42, 43, 44, 45, 46]
ZZ_SCALES = [0.05, 0.15, 0.3, 0.6, 1.0]
N_PILOT = 100_000
N_CALIB = 15
N_TEST_CLEAN = 15
N_TEST_TAMPERED = 15
EXPERIMENT_ID = "paper13_exp54_naive_diagnostic_baseline_v1"


def _paired_scores(rho_true, jac_before, r_dc_before_offdiag, g_before_offdiag, pool, n_pilot, rng):
    """One pilot draw, two scores from the SAME reconstructed state."""
    pilot = simulate_pilot(rho_true, pool.all_matrices, n_pilot, rng)
    route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
    g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)

    r_dc_hat_offdiag = compute_R_manifold(
        route_a_out.rho_hat, jac_before, pool, g_coef_override=g_hat_coef
    ).r_DC_coef[pool.offdiag_idx]
    score_manifold = float(np.linalg.norm(r_dc_hat_offdiag - r_dc_before_offdiag))

    g_hat_offdiag = g_hat_coef[pool.offdiag_idx]
    score_naive = float(np.linalg.norm(g_hat_offdiag - g_before_offdiag))

    return score_manifold, score_naive


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

        # --- calibration batch (clean only, not scored/reported) ---
        for rep in range(n_calib):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"calib_{rep}", "pilot")
            rng = np.random.default_rng(s)
            _paired_scores(rho_before, jac_before, r_dc_before_offdiag, g_before_offdiag, pool, N_PILOT, rng)

        # --- clean, TEST-set draws (label 0) ---
        for rep in range(n_test_clean):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"testclean_{rep}", "pilot")
            rng = np.random.default_rng(s)
            sm, sn = _paired_scores(rho_before, jac_before, r_dc_before_offdiag, g_before_offdiag, pool, N_PILOT, rng)
            score_rows.append(dict(seed=seed, condition="clean", label=0, score_manifold=sm, score_naive=sn))
        print(f"seed={seed}: clean calibration ({n_calib}) + test ({n_test_clean}) done", flush=True)

        # --- tampered, TEST-set draws (label 1), ring-ZZ magnitude sweep ---
        for scale in zz_scales:
            ts = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=N_QUBITS, layer=8,
                             ring_pairs=ring_pairs, n_a=N_A, x_ref=x_ct[:1], final_ca=float("nan"), phis=phis)
            intervention_fn = lambda flat, phis=phis, ring_pairs=ring_pairs, scale=scale: apply_zz_scaled(
                flat, N_QUBITS, phis, ring_pairs, scale)
            rho_after, jac_after = rho_A_and_jacobian(ts, "after", intervention_fn=intervention_fn)
            for rep in range(n_test_tampered):
                s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"zz{scale}_{rep}", "pilot")
                rng = np.random.default_rng(s)
                sm, sn = _paired_scores(rho_after, jac_before, r_dc_before_offdiag, g_before_offdiag, pool, N_PILOT, rng)
                score_rows.append(dict(seed=seed, condition=f"zz_scale_{scale}", label=1,
                                        score_manifold=sm, score_naive=sn))
            print(f"  zz scale={scale}: {n_test_tampered} tampered draws done", flush=True)

        # --- tampered, TEST-set draws (label 1), Z-only mechanism ---
        ts_z = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=N_QUBITS, layer=8,
                           ring_pairs=ring_pairs, n_a=N_A, x_ref=x_ct[:1], final_ca=float("nan"), phis=phis)
        z_thetas = list(np.random.RandomState(seed + 90001).uniform(-1.5, 1.5, size=N_QUBITS))
        z_fn = lambda flat, z_thetas=z_thetas: apply_single_z_null_unitary(flat, N_QUBITS, z_thetas)
        rho_z, jac_z = rho_A_and_jacobian(ts_z, "after", intervention_fn=z_fn)
        for rep in range(n_test_tampered):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"zonly_{rep}", "pilot")
            rng = np.random.default_rng(s)
            sm, sn = _paired_scores(rho_z, jac_before, r_dc_before_offdiag, g_before_offdiag, pool, N_PILOT, rng)
            score_rows.append(dict(seed=seed, condition="z_only", label=1, score_manifold=sm, score_naive=sn))
        print(f"  z_only: {n_test_tampered} tampered draws done", flush=True)

        _write_csv(RESULTS_DIR / "naive_diagnostic_baseline_scores.csv", score_rows)

    print(f"\nDone collecting {len(score_rows)} scored trials.")
    _summarize(score_rows)


def _summarize(rows):
    labels = np.array([r["label"] for r in rows])
    for key, name in [("score_manifold", "manifold-restricted (existing, r_DC)"),
                       ("score_naive", "naive raw off-diagonal Pauli discrepancy")]:
        scores = np.array([r[key] for r in rows])
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresh = roc_curve(labels, scores)

        def tpr_at_fpr(target_fpr):
            idx = np.searchsorted(fpr, target_fpr, side="right") - 1
            idx = max(idx, 0)
            return float(tpr[idx])

        print(f"\n=== Pooled end-to-end detection -- {name} ===")
        print(f"  ROC-AUC = {auc:.4f}")
        print(f"  TPR@1%FPR = {tpr_at_fpr(0.01)*100:.1f}%")
        print(f"  TPR@5%FPR = {tpr_at_fpr(0.05)*100:.1f}%")

        print(f"  Per-condition AUC:")
        clean_scores = scores[labels == 0]
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
