"""Experiment 52 (README Follow-up 65; guide/model_restricted_observability_
theory_revision_plan.md Section 4). The completion selection rule used
throughout this project is `j* = argmax_j |r_DC_coef[j]|^2` -- the
NUMERATOR of the true one-step blind-amplitude reduction. The guide
document raises a real point: adding candidate P_j to the measured set
only removes the component of r_DC ALONG q_j = P_N P_j (P_j's own
projection onto the current null space N), so the EXACT reduction in
gamma_D_C^2 from adding P_j is

    Delta(gamma_D_C^2)_j = |<r_DC, P_j>|^2 / ||P_N P_j||^2

not the bare numerator |<r_DC, P_j>|^2 -- these coincide only if ||P_N
P_j|| happens to be constant across candidates, which is not guaranteed.

This experiment checks, for all 10 real BloodMNIST states already used
throughout this project (no retraining, reusing persisted artifacts):
(1) whether ||P_N P_j|| is actually constant across the 240 offdiag
candidates (it is not -- reported range); (2) whether the numerator-only
argmax and the properly normalized argmax ever pick a DIFFERENT j*; (3)
when they disagree, how much gamma_D_C is actually left on the table by
the simpler (numerator-only) rule, in absolute and relative terms.

Usage:
    python run_experiment52_normalized_completion_score.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.restricted_observability import compute_R_manifold, pauli_coefficients

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
SEEDS = [42, 43, 44, 45, 46]
STAGES = ["before", "after"]


def _check_state(rho, jac, pool, label):
    res = compute_R_manifold(rho, jac, pool)
    r_dc = res.r_DC_coef
    baseline = res.gamma_D_C

    jac_moved = np.moveaxis(jac, -1, 0)
    jac_coef = pauli_coefficients(jac_moved, pool.all_matrices).T
    U, S, _ = np.linalg.svd(jac_coef, full_matrices=False)
    rank = int((S > 1e-8 * S.max()).sum())
    E = U[:, :rank]
    A = E[list(pool.diag_idx), :]
    _, Sa, Vha = np.linalg.svd(A, full_matrices=True)
    rank_a = int((Sa > 1e-6 * max(Sa.max(), 1e-300)).sum())
    N_mat = Vha.T[:, rank_a:]
    P_N = N_mat @ N_mat.T

    offdiag = np.array(pool.offdiag_idx)
    E_rows = E[offdiag, :]
    q_tangent = E_rows @ P_N.T
    norms = np.linalg.norm(q_tangent, axis=1)

    numerator_only = r_dc[offdiag] ** 2
    normalized = np.where(norms > 1e-8, numerator_only / np.maximum(norms ** 2, 1e-300), 0.0)
    j_num = int(offdiag[np.argmax(numerator_only)])
    j_norm = int(offdiag[np.argmax(normalized)])
    agree = j_num == j_norm

    eval_num = compute_R_manifold(rho, jac, pool, extra_measured_idx=[j_num]).gamma_D_C
    eval_norm = compute_R_manifold(rho, jac, pool, extra_measured_idx=[j_norm]).gamma_D_C

    row = dict(
        state=label, norm_q_min=float(norms.min()), norm_q_max=float(norms.max()),
        norm_q_std=float(norms.std()), j_numerator=j_num, j_normalized=j_norm, agree=agree,
        baseline_gamma_D_C=baseline,
        gamma_D_C_after_numerator=eval_num, gamma_D_C_after_normalized=eval_norm,
        reduction_numerator=baseline - eval_num, reduction_normalized=baseline - eval_norm,
        suboptimality_abs=eval_num - eval_norm,
        suboptimality_pct_of_baseline=(eval_num - eval_norm) / baseline * 100 if baseline > 1e-9 else float("nan"),
    )
    print(f"{label:30s} normsQ=[{norms.min():.3f},{norms.max():.3f}] "
          f"j_num={j_num:3d}({pool.all_labels[j_num]}) j_norm={j_norm:3d}({pool.all_labels[j_norm]}) "
          f"agree={agree} suboptimality={row['suboptimality_pct_of_baseline']:.3f}pp", flush=True)
    return row


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS

    rows = []
    for seed in seeds:
        for stage in STAGES:
            art = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_{stage}.npz")
            row = _check_state(art["rho_A"], art["jac"], pool, f"{DATASET}_seed{seed}_{stage}")
            rows.append(row)
            _write_csv(RESULTS_DIR / "normalized_completion_score_check.csv", rows)

    n_disagree = sum(1 for r in rows if not r["agree"])
    print(f"\nDone. {len(rows)} states checked, {n_disagree} disagreement(s).")
    if n_disagree:
        max_subopt = max(r["suboptimality_pct_of_baseline"] for r in rows if not r["agree"])
        print(f"Max sub-optimality among disagreements: {max_subopt:.3f} percentage points of baseline gamma_D_C.")


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
