"""Experiment 28: is Follow-up 37's weak single-observable multi-sample
completion a TUNING problem (robust_avg/robust_worst were both proxy
criteria, based on the pre-completion r_DC magnitude, not the actual
post-completion objective) or a FUNDAMENTAL one? Directly answers the
user's question "是否通过调参解决" (can this be fixed by better tuning).

Brute-force: for each of the 10 real BloodMNIST states, evaluate ALL 240
offdiag candidates directly on the objective that actually matters
(worst-case remaining gamma_D_C across the same 5 reference samples
Experiment 27 used), not a proxy. j*_oracle = the candidate that ACTUALLY
minimizes this. If j*_oracle does meaningfully better than Follow-up 37's
j*_robust_avg, the earlier result was a tunable proxy-selection problem.
If j*_oracle is similarly weak, no selection criterion can fix it --
it's a structural property of how spread out the per-sample blind
directions are, not a tuning problem.

Cheap: reuses the same rho_A/jac per (state, sample) Experiment 27
already computed (recomputed here since Experiment 27 didn't persist the
per-sample arrays), but the actual 240-candidate sweep uses
compute_R_manifold's extra_measured_idx path, which needs no fresh
autograd -- fast matrix ops only.

Usage:
    python run_experiment28_brute_force_oracle_completion.py [--smoke]
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
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
STAGES = ["before", "after"]
N_SAMPLES = 5


def run(smoke: bool):
    pool = build_pool(N_A)
    out_path = RESULTS_DIR / "brute_force_oracle_completion.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    n_candidates = 20 if smoke else len(pool.offdiag_idx)

    # also read Follow-up 37's own numbers for direct side-by-side comparison
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

            worst_after = np.full(n_candidates, np.inf)
            mean_after = np.zeros(n_candidates)
            for j_local in range(n_candidates):
                j_global = pool.offdiag_idx[j_local]
                after_vals = [compute_R_manifold(rho_list[k], jac_list[k], pool,
                                                   extra_measured_idx=[j_global]).gamma_D_C
                              for k in range(N_SAMPLES)]
                worst_after[j_local] = max(after_vals)
                mean_after[j_local] = float(np.mean(after_vals))

            j_oracle_worst = int(np.argmin(worst_after))
            j_oracle_mean = int(np.argmin(mean_after))
            oracle_worst_val = float(worst_after[j_oracle_worst])
            oracle_mean_val = float(mean_after[j_oracle_mean])

            fu37_robust_avg_worst = float(followup37[state_id]["gamma_D_C_after_robust_avg_worst"])
            fu37_ambient_worst = float(followup37[state_id]["gamma_D_C_after_ambient_worst"])

            reduction_oracle_pct = (before_worst - oracle_worst_val) / before_worst * 100
            reduction_fu37_pct = (before_worst - fu37_robust_avg_worst) / before_worst * 100

            row = dict(
                state_id=state_id, seed=seed, stage=stage, n_candidates_searched=n_candidates,
                before_worst=before_worst,
                j_oracle_worst_label=pool.offdiag_labels[j_oracle_worst],
                oracle_worst_after=oracle_worst_val,
                oracle_reduction_pct=reduction_oracle_pct,
                fu37_robust_avg_worst_after=fu37_robust_avg_worst,
                fu37_robust_avg_reduction_pct=reduction_fu37_pct,
                fu37_ambient_worst_after=fu37_ambient_worst,
                oracle_beats_robust_avg=(oracle_worst_val < fu37_robust_avg_worst - 1e-6),
            )
            rows.append(row)
            elapsed = time.time() - t0
            print(f"[{len(rows)}/{len(seeds)*len(STAGES)}] {state_id}: before_worst={before_worst:.3f} | "
                  f"TRUE ORACLE best candidate={pool.offdiag_labels[j_oracle_worst]} "
                  f"after={oracle_worst_val:.3f} (reduction={reduction_oracle_pct:.1f}%) | "
                  f"Follow-up37 robust_avg reduction={reduction_fu37_pct:.1f}% "
                  f"(elapsed {elapsed:.0f}s)", flush=True)
            _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path} ({time.time()-t0:.0f}s total).")
    _summarize(rows)
    return rows


def _summarize(rows):
    if not rows:
        return
    oracle_red = [r["oracle_reduction_pct"] for r in rows]
    fu37_red = [r["fu37_robust_avg_reduction_pct"] for r in rows]
    n_beats = sum(1 for r in rows if r["oracle_beats_robust_avg"])
    print(f"  TRUE ORACLE (searched {rows[0]['n_candidates_searched']} candidates) reduction%: "
          f"mean={np.mean(oracle_red):.2f} median={np.median(oracle_red):.2f} "
          f"min={np.min(oracle_red):.2f} max={np.max(oracle_red):.2f}")
    print(f"  Follow-up 37 robust_avg reduction%:                        "
          f"mean={np.mean(fu37_red):.2f} median={np.median(fu37_red):.2f} "
          f"min={np.min(fu37_red):.2f} max={np.max(fu37_red):.2f}")
    print(f"  oracle strictly beats robust_avg in {n_beats}/{len(rows)} states "
          f"(gap = how much a smarter, non-proxy selection criterion would have bought)")


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
