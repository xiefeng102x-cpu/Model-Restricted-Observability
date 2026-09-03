"""Experiment 30 (README Follow-up 40, direction "2" from the post-
Follow-up-39 reframing): does a SMALLER target subsystem make multi-
sample completion easier?

BloodMNIST's n_a=4 gives d=16, ambient dimension d^2-1=255 -- Follow-up
39 showed even a 3-observable greedy budget only reaches ~8% mean
worst-case reduction across 5 reference samples on that space. n_a=3
gives d=8, ambient=63 (4x smaller) -- a single observable now covers a
much larger FRACTION of the space. If the multi-sample-instability
bottleneck (Follow-up 36-39) is really about ambient-space SIZE, not
something structurally worse about this specific classifier, completion
should scale up meaningfully easier here. If it's similarly hard, the
bottleneck is not primarily a dimension-counting effect.

Reuses the SAME persisted theta (Experiment 25's results/manifold_
artifacts/*.npz) -- no retraining, just a different bipartition (first 3
of 8 qubits instead of first 4) of the SAME trained 8-qubit state.

Two parts, mirroring Follow-up 34 and Follow-up 39 at the smaller size:
  (a) single-x_ref ambient R_manifold, all 10 states (like Experiment 23)
  (b) 5-sample greedy multi-observable completion, budget=1,2,3 (like
      Experiment 29)

Usage:
    python run_experiment30_smaller_subsystem_n_a3.py [--smoke]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.theta_jacobian import ThetaState, rho_A_and_jacobian, di
from manifold.restricted_observability import compute_R_manifold

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 3
SEEDS = [42, 43, 44, 45, 46]
STAGES = ["before", "after"]
N_SAMPLES = 5
MAX_BUDGET = 3


def _worst_case(rho_list, jac_list, pool, selected_global_idx):
    vals = [compute_R_manifold(rho_list[k], jac_list[k], pool,
                                extra_measured_idx=selected_global_idx).gamma_D_C
            for k in range(len(rho_list))]
    return max(vals)


def greedy_search(rho_list, jac_list, pool, max_budget):
    selected_global = []
    trace = []
    remaining = list(range(len(pool.offdiag_idx)))
    for _ in range(max_budget):
        best_j, best_val = None, np.inf
        for j_local in remaining:
            j_global = pool.offdiag_idx[j_local]
            val = _worst_case(rho_list, jac_list, pool, selected_global + [j_global])
            if val < best_val:
                best_val, best_j = val, j_local
        selected_global.append(pool.offdiag_idx[best_j])
        remaining.remove(best_j)
        trace.append((best_j, best_val))
    return trace


def run(smoke: bool):
    pool = build_pool(N_A)
    out_ambient = RESULTS_DIR / "n_a3_ambient_R_manifold.csv"
    out_multi = RESULTS_DIR / "n_a3_greedy_multi_observable.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    n_samples = 2 if smoke else N_SAMPLES
    max_budget = 2 if smoke else MAX_BUDGET

    ambient_rows, multi_rows = [], []
    t0 = time.time()
    for seed in seeds:
        cfg = di.DATASETS[DATASET]
        x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])

        for stage in STAGES:
            state_id = f"{DATASET}_seed{seed}_{stage}"
            art = np.load(ARTIFACTS_DIR / f"{state_id}.npz")
            theta0 = torch.tensor(art["theta"], dtype=torch.float64, requires_grad=True)

            phis = None
            if stage == "after":
                rng_phi = np.random.RandomState(seed)
                phis = list(rng_phi.uniform(-1.5, 1.5, size=8))

            # (a) single-x_ref ambient R_manifold at n_a=3
            ts0 = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=8, layer=8,
                              ring_pairs=[(q, (q + 1) % 8) for q in range(8)], n_a=N_A,
                              x_ref=x_ct[:1], final_ca=float("nan"), phis=phis)
            rho0, jac0 = rho_A_and_jacobian(ts0, stage)
            res0 = compute_R_manifold(rho0, jac0, pool)
            ambient_rows.append(dict(state_id=state_id, seed=seed, stage=stage,
                                      gamma_D=res0.gamma_D, gamma_D_C=res0.gamma_D_C,
                                      R_manifold=res0.R_manifold, tangent_rank=res0.tangent_rank,
                                      null_dim=res0.null_dim, ambient_dim=pool.d ** 2 - 1))

            # (b) 5-sample greedy multi-observable completion at n_a=3
            rho_list, jac_list = [rho0], [jac0]
            for k in range(1, n_samples):
                ts_k = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=8, layer=8,
                                   ring_pairs=[(q, (q + 1) % 8) for q in range(8)], n_a=N_A,
                                   x_ref=x_ct[k:k + 1], final_ca=float("nan"), phis=phis)
                rho_k, jac_k = rho_A_and_jacobian(ts_k, stage)
                rho_list.append(rho_k); jac_list.append(jac_k)

            before_worst = max(compute_R_manifold(rho_list[k], jac_list[k], pool).gamma_D_C
                                for k in range(n_samples))
            trace = greedy_search(rho_list, jac_list, pool, max_budget)

            row = dict(state_id=state_id, seed=seed, stage=stage, before_worst=before_worst,
                       ambient_dim=pool.d ** 2 - 1)
            for step, (j_local, val) in enumerate(trace, start=1):
                row[f"budget{step}_label"] = pool.offdiag_labels[j_local]
                row[f"budget{step}_worst_after"] = val
                row[f"budget{step}_cumulative_reduction_pct"] = (before_worst - val) / before_worst * 100
            multi_rows.append(row)

            elapsed = time.time() - t0
            print(f"[{len(ambient_rows)}/{len(seeds)*len(STAGES)}] {state_id}: "
                  f"ambient R_manifold={res0.R_manifold:.4f} (tangent_rank={res0.tangent_rank}/128) | "
                  f"5-sample before_worst={before_worst:.3f} -> cumulative reduction% by budget: "
                  f"{[round(row[f'budget{s}_cumulative_reduction_pct'],1) for s in range(1, max_budget+1)]} "
                  f"(elapsed {elapsed:.0f}s)", flush=True)
            _write_csv(out_ambient, ambient_rows)
            _write_csv(out_multi, multi_rows)

    print(f"\nDone. {time.time()-t0:.0f}s total.")
    _summarize(ambient_rows, multi_rows, max_budget)
    return ambient_rows, multi_rows


def _summarize(ambient_rows, multi_rows, max_budget):
    if ambient_rows:
        vals = [r["R_manifold"] for r in ambient_rows]
        print(f"\n=== n_a=3 ambient R_manifold: median={np.median(vals):.4f} mean={np.mean(vals):.4f} "
              f"min={np.min(vals):.4f} max={np.max(vals):.4f} (n={len(vals)}) ===")
    if multi_rows:
        print("=== n_a=3, 5-sample greedy completion ===")
        for step in range(1, max_budget + 1):
            v = [r[f"budget{step}_cumulative_reduction_pct"] for r in multi_rows]
            print(f"  budget={step}: mean={np.mean(v):.2f}% median={np.median(v):.2f}%")


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
