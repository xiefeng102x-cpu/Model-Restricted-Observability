"""Experiment 24: R_manifold on this repository's mvp133 VQE checkpoints (README
Follow-up 34 continuation) -- a zero-retraining-cost stress test of the
manifold-restricted-observability machinery at a genuinely different
(qubit count, param count) regime than this repository's two classifier configs.

n_qubits=6, n_layers=6 -> 72 params, vs. MNIST's 120 (5q/12L) and
BloodMNIST's 128 (8q/8L). At n_a=4 (ambient dim 255), 72 params CANNOT
span the full ambient space by counting alone -- unlike both this repository
configs, where the tangent space saturated exactly (R_manifold=1.0000,
see Experiment 23's mnist_seed42 result). n_a=3 (natural symmetric half,
ambient dim 63) is also reported for comparison, though 72 > 63 there too
so it may still saturate.

No "task"/data here (VQE ground-state finding, not classification), so
there is no canonical target-subsystem choice -- both n_a values are
reported as a mechanistic check of the R_manifold machinery, not offered
as a second classifier-grade case study.

Usage:
    python run_experiment24_mvp133_vqe_R_manifold.py [--smoke]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.mvp133_vqe_checkpoints import load_theta, rho_A_and_jacobian, self_test as vqe_self_test
from manifold.restricted_observability import compute_R_manifold

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEEDS = list(range(10))
STAGES = ["midtrain", "cluster"]
N_A_VALUES = [3, 4]


def run(smoke: bool):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "mvp133_vqe_R_manifold.csv"

    vqe_self_test()

    seeds = SEEDS[:2] if smoke else SEEDS
    pools = {a: build_pool(a) for a in N_A_VALUES}

    rows = []
    t0 = time.time()
    for seed in seeds:
        for stage in STAGES:
            ts = load_theta(seed, stage)
            for n_a in N_A_VALUES:
                pool = pools[n_a]
                rho_A, jac = rho_A_and_jacobian(ts, n_a)
                res = compute_R_manifold(rho_A, jac, pool)
                ambient_dim = pool.d ** 2 - 1
                rows.append(dict(
                    seed=seed, stage=stage, n_a=n_a, d=pool.d, ambient_dim=ambient_dim,
                    n_params=int(ts.theta.numel()), final_energy=ts.final_energy,
                    gamma_D=res.gamma_D, gamma_D_C=res.gamma_D_C, R_manifold=res.R_manifold,
                    tangent_rank=res.tangent_rank, null_dim=res.null_dim,
                    params_lt_ambient=int(ts.theta.numel()) < ambient_dim,
                ))
                print(f"seed={seed} stage={stage:>9} n_a={n_a} (ambient={ambient_dim}, "
                      f"params={int(ts.theta.numel())}): R_manifold={res.R_manifold:.4f} "
                      f"tangent_rank={res.tangent_rank} null_dim={res.null_dim}", flush=True)
            _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path} ({time.time()-t0:.1f}s).")
    _summarize(rows)
    return rows


def _summarize(rows):
    print("\n=== R_manifold summary by n_a ===")
    for n_a in N_A_VALUES:
        vals = [r["R_manifold"] for r in rows if r["n_a"] == n_a and r["R_manifold"] == r["R_manifold"]]
        if not vals:
            continue
        ambient = n_a and (2 ** n_a) ** 2 - 1
        print(f"  n_a={n_a} (ambient_dim={ambient}): n={len(vals)} median={np.median(vals):.4f} "
              f"mean={np.mean(vals):.4f} min={np.min(vals):.4f} max={np.max(vals):.4f} "
              f"n_exactly_1={sum(1 for v in vals if v > 0.9999)}/{len(vals)}")


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
