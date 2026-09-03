"""Experiment 23: manifold-restricted local observability (this project's
next_stage_operational_observability_research_guide.md Priority-1
direction, Section 2.2-2.4; README Follow-up 34).

For each of this repository's 10 real trained classifiers (5 MNIST + 5 BloodMNIST
seeds) x {before, after the ZZ-null intervention}: reconstructs theta
(not persisted anywhere -- see manifold/theta_jacobian.py), computes the
tangent-space Jacobian d(rho_A)/d(theta), and the ratio

    R_manifold = gamma_{D,C} / gamma_D

where gamma_D is the ALREADY-established ambient blind-gradient amplitude
(oracle.entropy_target.score_candidates's gamma0, also already saved in
results/qmlreal_oracle_truth.csv's gamma0_true column) and gamma_{D,C} is
its manifold-restricted analogue (manifold/restricted_observability.py).

Two correctness checks are run per state, not just trusted from the
module self-tests:
  1. reconstructed final_ca must match the ORIGINAL saved .npz's final_ca
     (confirms theta reconstruction is exact, i.e. training is genuinely
     deterministic end to end, not just deterministic "in principle");
  2. reconstructed rho_A must match the ORIGINAL saved rho_A to high
     precision (confirms x_ref/model reconstruction, not just the
     downstream scalar CA, is exact) -- and gamma_D computed from the
     freshly-reconstructed rho_A must match qmlreal_oracle_truth.csv's
     already-published gamma0_true for that exact state_id.

Guide's own GO/NO-GO criterion (Section "GO-1"): median R_manifold <= 0.5,
stable across seeds -> manifold restriction becomes the paper's headline;
R_manifold ~= 1 -> pivot main line toward the phase-diagram framing
instead. Reported honestly either way.

Usage:
    python run_experiment23_manifold_restricted_observability.py [--smoke]
    python run_experiment23_manifold_restricted_observability.py --datasets mnist
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from oracle.entropy_target import score_candidates
from manifold.theta_jacobian import reconstruct_theta, rho_A_and_jacobian
from manifold.restricted_observability import compute_R_manifold

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SAVED_STATES_DIR = Path(__file__).resolve().parent / "data" / "saved_states"

SEEDS_BY_DATASET = {"mnist": [42, 43, 44, 45, 46], "bloodmnist": [42, 43, 44, 45, 46]}


def run(datasets: list, seeds_limit: int, stages: list, smoke: bool):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "manifold_restricted_observability.csv"

    rows = []
    t0 = time.time()
    n_done = 0
    n_total = sum(min(len(SEEDS_BY_DATASET[ds]), seeds_limit) for ds in datasets) * len(stages)

    for dataset_name in datasets:
        seeds = SEEDS_BY_DATASET[dataset_name][:seeds_limit]
        for seed in seeds:
            npz_path = SAVED_STATES_DIR / f"{dataset_name}_seed{seed}.npz"
            saved = np.load(npz_path, allow_pickle=True)

            t_train0 = time.time()
            ts = reconstruct_theta(dataset_name, seed)
            train_time = time.time() - t_train0

            saved_ca = float(saved["final_ca"])
            ca_match = abs(ts.final_ca - saved_ca) < 1e-6
            n_a = int(saved["n_a"])
            assert n_a == ts.n_a, f"{dataset_name}/seed{seed}: n_a mismatch {n_a} vs {ts.n_a}"
            pool = build_pool(n_a)

            for stage in stages:
                rho_A, jac = rho_A_and_jacobian(ts, stage)
                saved_rho = saved[f"rho_{stage}"]
                rho_match_err = float(np.abs(rho_A - saved_rho).max())

                oracle_gamma0 = score_candidates(rho_A, pool.offdiag_matrices).gamma0
                res = compute_R_manifold(rho_A, jac, pool)
                gamma_D_cross_err = abs(res.gamma_D - oracle_gamma0)

                n_done += 1
                elapsed = time.time() - t0
                print(f"[{n_done}/{n_total}] {dataset_name}_seed{seed}_{stage}: "
                      f"train_time={train_time:.0f}s ca_match={ca_match} "
                      f"rho_match_err={rho_match_err:.2e} "
                      f"gamma_D={res.gamma_D:.4f} gamma_D_C={res.gamma_D_C:.4f} "
                      f"R_manifold={res.R_manifold:.4f} "
                      f"tangent_rank={res.tangent_rank}/{ts.theta.numel()} null_dim={res.null_dim} "
                      f"(elapsed {elapsed:.0f}s)", flush=True)

                rows.append(dict(
                    state_id=f"{dataset_name}_seed{seed}_{stage}", dataset=dataset_name,
                    seed=seed, stage=stage, a=n_a, d=pool.d, n_params=ts.theta.numel(),
                    final_ca=ts.final_ca, saved_final_ca=saved_ca, ca_match=ca_match,
                    rho_match_max_abs_err=rho_match_err,
                    gamma_D=res.gamma_D, gamma_D_oracle_crosscheck=oracle_gamma0,
                    gamma_D_crosscheck_err=gamma_D_cross_err,
                    gamma_D_C=res.gamma_D_C, R_manifold=res.R_manifold,
                    tangent_rank=res.tangent_rank, null_dim=res.null_dim,
                    train_time_s=train_time,
                ))
                _write_csv(out_path, rows)

            if smoke:
                break
        if smoke:
            break

    print(f"\nDone. {len(rows)} state-rows written to {out_path} ({time.time()-t0:.0f}s total).")
    _summarize(rows)
    return rows


def _summarize(rows):
    if not rows:
        return
    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r["R_manifold"])
    print("\n=== R_manifold summary ===")
    all_vals = []
    for ds, vals in by_dataset.items():
        vals = [v for v in vals if v == v]  # drop NaN
        all_vals.extend(vals)
        if vals:
            print(f"  {ds}: n={len(vals)} median={np.median(vals):.4f} "
                  f"mean={np.mean(vals):.4f} min={np.min(vals):.4f} max={np.max(vals):.4f}")
    if all_vals:
        med = np.median(all_vals)
        print(f"  ALL: n={len(all_vals)} median={med:.4f}")
        if med <= 0.5:
            print(f"  -> GO-1 candidate (median R_manifold={med:.4f} <= 0.5): manifold "
                  f"restriction may become the paper's headline, per the guide's own criterion "
                  f"-- check per-seed stability before committing.")
        else:
            print(f"  -> median R_manifold={med:.4f}, NOT clearly < 1: per the guide's own "
                  f"fallback, consider pivoting the main line toward the phase-diagram framing "
                  f"instead of manifold restriction.")

    bad_ca = [r for r in rows if not r["ca_match"]]
    bad_rho = [r for r in rows if r["rho_match_max_abs_err"] > 1e-6]
    bad_gamma = [r for r in rows if r["gamma_D_crosscheck_err"] > 1e-6]
    print(f"\nCorrectness checks: {len(rows)-len(bad_ca)}/{len(rows)} exact CA match, "
          f"{len(rows)-len(bad_rho)}/{len(rows)} rho_A match (<1e-6), "
          f"{len(rows)-len(bad_gamma)}/{len(rows)} gamma_D oracle cross-check match (<1e-6).")
    if bad_ca or bad_rho or bad_gamma:
        print("  WARNING: some states failed a correctness check -- see CSV for details, "
              "do not trust R_manifold for those rows without investigating.")


def _write_csv(path: Path, rows: list):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1 state only (mnist seed42, before)")
    parser.add_argument("--datasets", nargs="+", default=["mnist", "bloodmnist"])
    parser.add_argument("--seeds-limit", type=int, default=5)
    parser.add_argument("--stages", nargs="+", default=["before", "after"])
    args = parser.parse_args()
    if args.smoke:
        run(["mnist"], 1, ["before"], smoke=True)
    else:
        run(args.datasets, args.seeds_limit, args.stages, smoke=False)
