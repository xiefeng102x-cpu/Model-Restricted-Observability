"""Experiment 25: guide Theorem B in practice -- does knowing the QML
model's own manifold constraint change WHICH Pauli measurement you should
add next? (README Follow-up 35, continuing Follow-up 34's R_manifold
result.)

For each BloodMNIST state (the 5 seeds where R_manifold < 1 was real, not
the MNIST over-parametrized-saturation case where T_rho C = ambient space
and this question is trivial by construction): computes r_{D,C}, the
guide's Theorem B completion direction (the model-reachable-but-
measurement-invisible part of the diagnostic gradient -- see
manifold/restricted_observability.py's self-test check 5 for a full
numerical confirmation that r_DC is (a) exactly measurement-invisible,
(b) a real nonzero diagnostic-sensitive direction, and (c) that adding it
as ONE new observable drives the completed gamma_{D,C} to ~0, i.e.
Theorem B's claim holds, not just algebraically but checked against an
independent finite/augmented-measurement-map recomputation).

Then asks the practical question: among this project's actual candidate
Pauli pool (pool.offdiag_idx, the same C0 set every selector in this
project chooses from), which single candidate has the largest overlap
with r_{D,C} (a "manifold-aware" j*_C, by literally the same |a_j|^2
scoring formula as oracle.entropy_target.score_candidates, just with the
gradient replaced by r_DC_coef) -- and does it match the AMBIENT j*
oracle.entropy_target already computed and published in
results/qmlreal_oracle_truth.csv?

Also fixes a real gap from Experiment 23 (this repository's own established rule,
see memory: training scripts must persist trained weights, not just
scalar summaries, or follow-up mechanism analysis needs re-retraining --
which is exactly what happened here): persists theta/rho_A/jac/r_DC_coef
per state to results/manifold_artifacts/, so this specific retraining
cost (~90 min for 5 BloodMNIST seeds) is paid at most once more.

Usage:
    python run_experiment25_manifold_aware_completion.py [--smoke]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.theta_jacobian import reconstruct_theta, rho_A_and_jacobian
from manifold.restricted_observability import compute_R_manifold

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
# qmlreal_oracle_truth.csv is produced by this repository's own run_experiment5
# (the ambient oracle j*/gamma0 baseline) -- lives there, not duplicated here.
ORACLE_CSV = Path(__file__).resolve().parent / "data" / "qmlreal_oracle_truth.csv"

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
STAGES = ["before", "after"]
DATASET = "bloodmnist"
N_A = 4


def _load_ambient_oracle():
    with open(ORACLE_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["state_id"]: r for r in rows}


def run(smoke: bool):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "manifold_aware_completion.csv"

    ambient = _load_ambient_oracle()
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS

    rows = []
    t0 = time.time()
    for seed in seeds:
        ts = reconstruct_theta(DATASET, seed)
        for stage in STAGES:
            state_id = f"{DATASET}_seed{seed}_{stage}"
            rho_A, jac = rho_A_and_jacobian(ts, stage)
            res = compute_R_manifold(rho_A, jac, pool)

            # Skip rewriting an already-cached artifact: other experiment
            # scripts may be concurrently READING this exact file for the
            # legacy seeds (42-46) right now, and an in-place np.savez
            # overwrite of a file with a concurrent reader is exactly what
            # produced a real BadZipFile/CRC race earlier this session --
            # only write files that don't exist yet (new seeds 47-51).
            art_path = ARTIFACTS_DIR / f"{state_id}.npz"
            if not art_path.exists():
                np.savez(art_path,
                         theta=ts.theta.detach().numpy(), rho_A=rho_A, jac=jac,
                         g_coef=res.g_coef, r_DC_coef=res.r_DC_coef,
                         gamma_D=res.gamma_D, gamma_D_C=res.gamma_D_C, R_manifold=res.R_manifold,
                         tangent_rank=res.tangent_rank, null_dim=res.null_dim)

            score_C = res.exact_score[pool.offdiag_idx]
            order_C = np.argsort(-score_C)
            j_star_C = int(order_C[0])
            margin_C = float(score_C[order_C[0]] - score_C[order_C[1]]) if len(order_C) > 1 else float("nan")

            amb = ambient[state_id]
            j_star_ambient = int(amb["oracle_j_star"])
            agree = j_star_C == j_star_ambient

            row = dict(
                state_id=state_id, seed=seed, stage=stage,
                R_manifold=res.R_manifold, gamma_D=res.gamma_D, gamma_D_C=res.gamma_D_C,
                j_star_C=j_star_C, j_star_C_pauli=pool.offdiag_labels[j_star_C], margin_C=margin_C,
                j_star_ambient=j_star_ambient, j_star_ambient_pauli=amb["oracle_pauli_string"],
                margin_ambient=float(amb["oracle_margin"]),
                agree=agree,
            )
            rows.append(row)
            print(f"[{len(rows)}/{len(seeds)*len(STAGES)}] {state_id}: R_manifold={res.R_manifold:.4f} "
                  f"j*_C={pool.offdiag_labels[j_star_C]} (margin={margin_C:.4f}) vs "
                  f"j*_ambient={amb['oracle_pauli_string']} (margin={float(amb['oracle_margin']):.4f}) "
                  f"-> {'AGREE' if agree else 'DISAGREE'}", flush=True)
            _write_csv(out_path, rows)

    n_agree = sum(1 for r in rows if r["agree"])
    print(f"\nDone. {len(rows)} rows written to {out_path} ({time.time()-t0:.0f}s total).")
    print(f"Manifold-aware j*_C vs ambient j*: agree {n_agree}/{len(rows)}.")
    return rows


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
