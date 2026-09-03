"""Experiment 32 (README Follow-up 43): the new direction's second
headline test, directly connecting back to this project's core backdoor-
relevant question (this repository's whole premise: a task-invariant perturbation
-- the ZZ-null-unitary -- that leaves the classifier's output exactly
unchanged but IS visible to an entropy-based diagnostic, IF you're
measuring the right thing).

Follow-up 42 showed j*_C flips 5/5 and j*_ambient flips 4/5 between
"before" and "after" for the SAME trained model and SAME input, only the
task-invisible perturbation applied. This asks the practically loaded
version of that observation: if you calibrate your extra measurement
based on CLEAN ("before") data -- the realistic deployment scenario, since
you don't know in advance which states are perturbed -- does it still
catch the blind spot once the ("after") perturbation has been applied, or
does the perturbation specifically evade a clean-calibrated completion?

Entirely reuses Experiment 25's persisted artifacts (results/manifold_
artifacts/*.npz: rho_A, jac, r_DC_coef for both before AND after, per
seed) -- NO retraining, NO fresh Jacobian computation, just the cheap
extra_measured_idx path on already-saved arrays.

For each of the 5 BloodMNIST seeds:
  - baseline_after   = gamma_D_C(after), no completion at all
  - clean_calibrated = gamma_D_C(after) after adding j*_C(before) as the
                        one new observable (the realistic "trained on
                        clean data, deployed on possibly-perturbed data"
                        scenario)
  - oracle_after     = gamma_D_C(after) after adding j*_C(after) itself
                        (the best possible completion, IF you already
                        knew this exact state was perturbed -- an
                        unrealistic upper bound, not a deployable
                        strategy, but the right reference point for "how
                        much of the achievable gain does clean-calibration
                        actually recover")

Usage:
    python run_experiment32_before_after_completion_evasion.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.restricted_observability import compute_R_manifold

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
SEEDS = [42, 43, 44, 45, 46]


def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def run():
    pool = build_pool(N_A)
    out_path = RESULTS_DIR / "before_after_completion_evasion.csv"

    rows = []
    for seed in SEEDS:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        art_after = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_after.npz")

        rho_before, jac_before = art_before["rho_A"], art_before["jac"]
        rho_after, jac_after = art_after["rho_A"], art_after["jac"]
        r_dc_before, r_dc_after = art_before["r_DC_coef"], art_after["r_DC_coef"]

        cos_sim = _cos(r_dc_before, r_dc_after)

        # Exact one-candidate reduction-maximizing rule (Section 4; README
        # Follow-up 96), replacing the numerator-only argmax(r_DC_coef**2)
        # used through Follow-up 95. Recomputed fresh here (cheap: rho_A/jac
        # are already persisted) since the .npz artifacts predate the
        # exact_score field.
        res_before_full = compute_R_manifold(rho_before, jac_before, pool)
        res_after_full = compute_R_manifold(rho_after, jac_after, pool)
        j_before_local = int(np.argmax(res_before_full.exact_score[pool.offdiag_idx]))
        j_after_local = int(np.argmax(res_after_full.exact_score[pool.offdiag_idx]))
        j_before_global = pool.offdiag_idx[j_before_local]
        j_after_global = pool.offdiag_idx[j_after_local]

        baseline_after = float(art_after["gamma_D_C"])
        clean_calibrated = compute_R_manifold(rho_after, jac_after, pool,
                                                extra_measured_idx=[j_before_global]).gamma_D_C
        oracle_after = compute_R_manifold(rho_after, jac_after, pool,
                                           extra_measured_idx=[j_after_global]).gamma_D_C

        oracle_reduction = baseline_after - oracle_after
        clean_reduction = baseline_after - clean_calibrated
        recovery_frac = clean_reduction / oracle_reduction if oracle_reduction > 1e-9 else float("nan")

        row = dict(seed=seed,
                    j_star_C_before=pool.offdiag_labels[j_before_local],
                    j_star_C_after=pool.offdiag_labels[j_after_local],
                    same_candidate=(j_before_local == j_after_local),
                    r_DC_cosine_similarity_before_after=cos_sim,
                    baseline_after_gamma_D_C=baseline_after,
                    clean_calibrated_gamma_D_C=clean_calibrated,
                    oracle_after_gamma_D_C=oracle_after,
                    oracle_reduction=oracle_reduction,
                    clean_calibrated_reduction=clean_reduction,
                    clean_calibration_recovery_fraction=recovery_frac)
        rows.append(row)
        print(f"seed={seed}: j*_C before={row['j_star_C_before']} after={row['j_star_C_after']} "
              f"(same={row['same_candidate']}) | cos(r_DC_before,r_DC_after)={cos_sim:.4f} | "
              f"baseline={baseline_after:.4f} clean_calibrated={clean_calibrated:.4f} "
              f"oracle_after={oracle_after:.4f} | clean recovers "
              f"{recovery_frac*100:.1f}% of the achievable reduction", flush=True)

    _write_csv(out_path, rows)
    print(f"\nDone. {len(rows)} rows written to {out_path}.")
    _summarize(rows)
    return rows


def _summarize(rows):
    cos_vals = [r["r_DC_cosine_similarity_before_after"] for r in rows]
    recov = [r["clean_calibration_recovery_fraction"] for r in rows]
    n_same = sum(1 for r in rows if r["same_candidate"])
    print(f"\n  j*_C(before) == j*_C(after): {n_same}/{len(rows)}")
    print(f"  cos(r_DC_before, r_DC_after): mean={np.mean(cos_vals):.4f} "
          f"median={np.median(cos_vals):.4f} (values: {[round(c,3) for c in cos_vals]})")
    print(f"  clean-calibration recovery fraction of achievable reduction: "
          f"mean={np.nanmean(recov)*100:.1f}% median={np.nanmedian(recov)*100:.1f}% "
          f"(values: {[round(r*100,1) for r in recov]})")


def _write_csv(path: Path, rows: list):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run()
