"""Experiment 31 (README Follow-up 41, direction "3" from the post-
Follow-up-39 reframing): are blind directions (r_DC) more similar WITHIN
a class than ACROSS classes?

Follow-up 36 showed r_DC changes substantially across reference samples
in general (0/40 j*_C stability), but all those samples were from the
SAME class (x_ct, "target-clean"). This checks whether that instability
is actually finer-grained than class identity (i.e. r_DC varies a lot
even within one class, so a class-conditional/adaptive completion
strategy -- direction "1" -- would not help much), or whether same-class
samples' r_DC directions cluster together more than different-class ones
(in which case a simple 2-branch class-adaptive scheme is worth trying
as a genuine, cheaper-than-per-sample form of input-adaptive completion).

Deliberately CHEAP and decisive before committing to the more expensive
adaptive-completion comparison: just computes r_DC for 5 x_ct (target-
clean) + 5 x_cnt (non-target-clean) reference samples per state (10 real
BloodMNIST states, theta reused from Experiment 25's persisted artifacts,
no retraining), and reports mean pairwise cosine similarity within each
group vs. across groups.

Usage:
    python run_experiment31_within_vs_across_class_structure.py [--smoke]
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
N_PER_GROUP = 5


def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def run(smoke: bool):
    pool = build_pool(N_A)
    out_path = RESULTS_DIR / "within_vs_across_class_similarity.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    n_per_group = 2 if smoke else N_PER_GROUP

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

            r_dc_ct, r_dc_cnt = [], []
            for k in range(n_per_group):
                for x_pool, store in ((x_ct, r_dc_ct), (x_cnt, r_dc_cnt)):
                    ts_k = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=8, layer=8,
                                       ring_pairs=[(q, (q + 1) % 8) for q in range(8)], n_a=N_A,
                                       x_ref=x_pool[k:k + 1], final_ca=float("nan"), phis=phis)
                    rho_k, jac_k = rho_A_and_jacobian(ts_k, stage)
                    res_k = compute_R_manifold(rho_k, jac_k, pool)
                    store.append(res_k.r_DC_coef)

            within_ct = [_cos(r_dc_ct[i], r_dc_ct[j])
                         for i in range(n_per_group) for j in range(i + 1, n_per_group)]
            within_cnt = [_cos(r_dc_cnt[i], r_dc_cnt[j])
                          for i in range(n_per_group) for j in range(i + 1, n_per_group)]
            across = [_cos(r_dc_ct[i], r_dc_cnt[j])
                      for i in range(n_per_group) for j in range(n_per_group)]

            row = dict(state_id=state_id, seed=seed, stage=stage,
                       within_ct_mean_cos=float(np.nanmean(within_ct)),
                       within_cnt_mean_cos=float(np.nanmean(within_cnt)),
                       across_mean_cos=float(np.nanmean(across)),
                       within_ct_abs_mean=float(np.nanmean(np.abs(within_ct))),
                       within_cnt_abs_mean=float(np.nanmean(np.abs(within_cnt))),
                       across_abs_mean=float(np.nanmean(np.abs(across))))
            rows.append(row)

            elapsed = time.time() - t0
            print(f"[{len(rows)}/{len(seeds)*len(STAGES)}] {state_id}: "
                  f"within_ct={row['within_ct_mean_cos']:.3f} within_cnt={row['within_cnt_mean_cos']:.3f} "
                  f"across={row['across_mean_cos']:.3f} | |cos| within_ct={row['within_ct_abs_mean']:.3f} "
                  f"within_cnt={row['within_cnt_abs_mean']:.3f} across={row['across_abs_mean']:.3f} "
                  f"(elapsed {elapsed:.0f}s)", flush=True)
            _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path} ({time.time()-t0:.0f}s total).")
    _summarize(rows)
    return rows


def _summarize(rows):
    if not rows:
        return
    for key in ["within_ct_abs_mean", "within_cnt_abs_mean", "across_abs_mean"]:
        vals = [r[key] for r in rows]
        print(f"  {key}: mean={np.mean(vals):.4f} median={np.median(vals):.4f}")
    within_avg = np.mean([(r["within_ct_abs_mean"] + r["within_cnt_abs_mean"]) / 2 for r in rows])
    across_avg = np.mean([r["across_abs_mean"] for r in rows])
    print(f"\n  Overall: mean|cos| WITHIN-class={within_avg:.4f} vs ACROSS-class={across_avg:.4f}")
    if within_avg > across_avg * 1.3:
        print("  -> WITHIN-class similarity notably higher: class-adaptive completion likely worth trying.")
    elif within_avg < across_avg * 0.8:
        print("  -> WITHIN-class similarity is LOWER than across-class (surprising) -- class is not the right adaptivity axis.")
    else:
        print("  -> No strong within/across-class separation -- r_DC instability is finer-grained than class identity; "
              "simple class-conditional adaptivity would likely not help much.")


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
