"""Experiment 29: Follow-up 38's own flagged next step -- Follow-up 28
showed a single observable (chosen in any way, even by exhaustive search)
caps out around ~5% worst-case-remaining-gamma_D_C reduction across 5
reference samples of the same trained model. Is the ceiling a 1-observable
artifact, or does it persist as you add more? Greedy forward search:
budget=1,2,3 real offdiag Pauli observables, added one at a time, each
step picking whichever REMAINING candidate most reduces the worst-case
gamma_D_C across the same 5 samples (given everything already selected).

Reuses the same per-(state,sample) rho_A/jac Experiment 27/28 computed
(recomputed here -- not persisted by either, same ~5 Jacobian calls per
state as before). The greedy search itself uses only the cheap
`extra_measured_idx` path (no fresh autograd).

Usage:
    python run_experiment29_greedy_multi_observable_budget.py [--smoke]
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
N_A = 4
SEEDS = [42, 43, 44, 45, 46]
STAGES = ["before", "after"]
N_SAMPLES = 5
MAX_BUDGET = 3


def _worst_case(rho_list, jac_list, pool, selected_global_idx):
    vals = [compute_R_manifold(rho_list[k], jac_list[k], pool,
                                extra_measured_idx=selected_global_idx).gamma_D_C
            for k in range(len(rho_list))]
    return max(vals)


def greedy_search(rho_list, jac_list, pool, max_budget, candidate_pool_local):
    selected_local = []
    selected_global = []
    trace = []
    remaining = list(candidate_pool_local)
    for step in range(max_budget):
        best_j, best_val = None, np.inf
        for j_local in remaining:
            j_global = pool.offdiag_idx[j_local]
            val = _worst_case(rho_list, jac_list, pool, selected_global + [j_global])
            if val < best_val:
                best_val, best_j = val, j_local
        selected_local.append(best_j)
        selected_global.append(pool.offdiag_idx[best_j])
        remaining.remove(best_j)
        trace.append((best_j, best_val))
    return selected_local, trace


def run(smoke: bool):
    pool = build_pool(N_A)
    out_path = RESULTS_DIR / "greedy_multi_observable_budget.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    max_budget = 2 if smoke else MAX_BUDGET
    candidate_pool_local = list(range(30)) if smoke else list(range(len(pool.offdiag_idx)))

    with open(RESULTS_DIR / "multi_sample_robust_completion.csv", newline="", encoding="utf-8") as f:
        followup37 = {r["state_id"]: r for r in csv.DictReader(f)}

    rows = []
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

            rho_list, jac_list = [], []
            for k in range(N_SAMPLES):
                ts_k = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=8, layer=8,
                                   ring_pairs=[(q, (q + 1) % 8) for q in range(8)], n_a=N_A,
                                   x_ref=x_ct[k:k + 1], final_ca=float("nan"), phis=phis)
                rho_k, jac_k = rho_A_and_jacobian(ts_k, stage)
                rho_list.append(rho_k); jac_list.append(jac_k)

            before_worst = float(followup37[state_id]["gamma_D_C_before_worst"])
            selected_local, trace = greedy_search(rho_list, jac_list, pool, max_budget, candidate_pool_local)

            row = dict(state_id=state_id, seed=seed, stage=stage, before_worst=before_worst)
            prev_val = before_worst
            labels = []
            for step, (j_local, val) in enumerate(trace, start=1):
                label = pool.offdiag_labels[j_local]
                labels.append(label)
                step_reduction_pct = (before_worst - val) / before_worst * 100
                marginal_pct = (prev_val - val) / before_worst * 100
                row[f"budget{step}_label"] = label
                row[f"budget{step}_worst_after"] = val
                row[f"budget{step}_cumulative_reduction_pct"] = step_reduction_pct
                row[f"budget{step}_marginal_reduction_pct"] = marginal_pct
                prev_val = val
            rows.append(row)

            elapsed = time.time() - t0
            print(f"[{len(rows)}/{len(seeds)*len(STAGES)}] {state_id}: before_worst={before_worst:.3f} | "
                  f"greedy picks={labels} | cumulative reduction% by budget: "
                  f"{[round(row[f'budget{s}_cumulative_reduction_pct'],1) for s in range(1, max_budget+1)]} "
                  f"(elapsed {elapsed:.0f}s)", flush=True)
            _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path} ({time.time()-t0:.0f}s total).")
    _summarize(rows, max_budget)
    return rows


def _summarize(rows, max_budget):
    if not rows:
        return
    print("\n=== Cumulative worst-case-remaining-gamma_D_C reduction by observable budget ===")
    for step in range(1, max_budget + 1):
        vals = [r[f"budget{step}_cumulative_reduction_pct"] for r in rows]
        print(f"  budget={step}: mean={np.mean(vals):.2f}% median={np.median(vals):.2f}% "
              f"min={np.min(vals):.2f}% max={np.max(vals):.2f}%")


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
