"""Experiment 38 (README Follow-up 49; manuscript Discussion item "P0-3
not yet done"): the first end-to-end clean-versus-tampered detection
score for this direction, with a proper ROC/AUC, not just completion-
recovery metrics.

Score (reuses Experiment 37's already-validated gate-discrepancy
construction unchanged, not a new formula invented for this experiment):
the same pilot draw already needed for adaptive recalibration gives a
pilot-estimated r_DC (via the trusted jac_before), compared against the
trusted clean-state r_DC. This is a single number per deployment, cheap,
and -- critically, argued in Experiment 37's own docstring -- NOT based
on the already-measured (diagonal, task) Pauli values, since those are
exactly what every task-invariant perturbation in this project preserves
by construction.

Calibration discipline: for each seed, an independent CALIBRATION batch
of clean-state pilot draws is kept entirely separate from the clean
draws used in the reported ROC/AUC ("test") -- the calibration batch is
what a real deployment would use to set an operating threshold; it is
never mixed into the reported detection statistics, honoring the guide's
explicit "calibration set and test set must be kept separate" requirement
even though ROC-AUC itself is threshold-free and would not numerically
change if this were skipped.

Tampered class: 5 magnitude levels of the primary ring-ZZ mechanism
(0.05, 0.15, 0.3, 0.6, 1.0 of the original scale) plus the Z-only
(non-entangling) mechanism at full magnitude -- 6 conditions x 5 seeds,
spanning both severity and mechanism type, per the guide's own P0-3
specification.

Usage:
    python run_experiment38_end_to_end_detection.py [--smoke]
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
from manifold.restricted_observability import compute_R_manifold
from seeds import derive_seed
from run_experiment35_multiple_perturbation_mechanisms import apply_zz_scaled, apply_single_z_null_unitary
from run_experiment37_gated_adaptive_audit import _r_dc_offdiag, _pilot_discrepancy

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
N_QUBITS = 8
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
ZZ_SCALES = [0.05, 0.15, 0.3, 0.6, 1.0]
N_PILOT = 100_000
N_CALIB = 15    # clean-only, threshold-setting, excluded from the reported ROC
N_TEST_CLEAN = 15   # clean, held out, included in the reported ROC (label 0)
N_TEST_TAMPERED = 15  # per tampered condition (label 1)
EXPERIMENT_ID = "paper13_exp38_e2e_detection_v1"


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
        r_dc_before_offdiag = _r_dc_offdiag(rho_before, jac_before, pool)

        cfg = di.DATASETS[DATASET]
        x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
        ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
        rng_phi = np.random.RandomState(seed)
        phis = list(rng_phi.uniform(-1.5, 1.5, size=N_QUBITS))

        # --- calibration batch (clean only, excluded from reported ROC) ---
        for rep in range(n_calib):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"calib_{rep}", "pilot")
            rng = np.random.default_rng(s)
            delta, _, _ = _pilot_discrepancy(rho_before, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
            # not stored in score_rows -- calibration-only, by design

        # --- clean, TEST-set draws (label 0) ---
        for rep in range(n_test_clean):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"testclean_{rep}", "pilot")
            rng = np.random.default_rng(s)
            delta, _, _ = _pilot_discrepancy(rho_before, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
            score_rows.append(dict(seed=seed, condition="clean", label=0, delta_pilot=delta))
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
                delta, _, _ = _pilot_discrepancy(rho_after, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
                score_rows.append(dict(seed=seed, condition=f"zz_scale_{scale}", label=1, delta_pilot=delta))
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
            delta, _, _ = _pilot_discrepancy(rho_z, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
            score_rows.append(dict(seed=seed, condition="z_only", label=1, delta_pilot=delta))
        print(f"  z_only: {n_test_tampered} tampered draws done", flush=True)

        _write_csv(RESULTS_DIR / "e2e_detection_scores.csv", score_rows)

    print(f"\nDone collecting {len(score_rows)} scored trials.")
    _summarize(score_rows)


def _summarize(rows):
    labels = np.array([r["label"] for r in rows])
    scores = np.array([r["delta_pilot"] for r in rows])
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresh = roc_curve(labels, scores)

    def tpr_at_fpr(target_fpr):
        idx = np.searchsorted(fpr, target_fpr, side="right") - 1
        idx = max(idx, 0)
        return float(tpr[idx])

    print(f"\n=== Pooled end-to-end detection (clean vs. all tampered conditions) ===")
    print(f"  n_clean={int((labels==0).sum())}, n_tampered={int((labels==1).sum())}")
    print(f"  ROC-AUC = {auc:.4f}")
    print(f"  TPR@1%FPR = {tpr_at_fpr(0.01)*100:.1f}%")
    print(f"  TPR@5%FPR = {tpr_at_fpr(0.05)*100:.1f}%")

    print("\n=== Per-condition AUC (each tampered condition vs. the pooled clean class) ===")
    clean_scores = scores[labels == 0]
    for cond in sorted(set(r["condition"] for r in rows if r["label"] == 1)):
        cond_scores = np.array([r["delta_pilot"] for r in rows if r["condition"] == cond])
        y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(cond_scores))])
        s = np.concatenate([clean_scores, cond_scores])
        try:
            cond_auc = roc_auc_score(y, s)
        except ValueError:
            cond_auc = float("nan")
        print(f"  {cond:>16}: AUC={cond_auc:.4f} (n={len(cond_scores)}, mean score={cond_scores.mean():.4f} "
              f"vs clean mean={clean_scores.mean():.4f})")


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
