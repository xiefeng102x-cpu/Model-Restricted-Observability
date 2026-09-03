"""Experiment 27: guide Theorem C in practice (README Follow-up 37,
continuing Follow-up 36's finding that j*_C is a per-input, not a global,
recommendation).

Instead of picking the single best measurement direction from ONE
reference sample x_ref (Follow-up 35's j*_C, shown in Follow-up 36 to
flip 0/40 times across alternative same-class samples), this asks the
practically-meaningful version of the question: given several
representative reference samples for the SAME trained model, is there a
SINGLE candidate Pauli direction that helps across all of them --
guide Theorem C's "does one added observable achieve a smaller universal
completion than the ambient one" question, made concrete and testable.

For each of the 10 real BloodMNIST states (theta reused from Experiment
25's persisted artifacts, no retraining): computes r_DC across 5 reference
samples (x_ct[0:1]..x_ct[4:5], the same samples Experiment 26 used),
derives two "robust" candidates --
  j*_robust_avg  = argmax_m mean_k (r_DC^(k)[m])^2      (best on average)
  j*_robust_worst = argmax_m min_k (r_DC^(k)[m])^2       (best worst-case)
-- and compares, for EACH of {j*_ambient, j*_robust_avg, j*_robust_worst},
how much adding it as ONE new observable (manifold/restricted_
observability.py's `extra_measured_idx`) reduces gamma_D_C across all 5
samples, reporting the WORST-CASE (least-helped) remaining gamma_D_C for
each candidate -- the practically relevant number if you can only budget
one new measurement and want it to work reasonably well no matter which
input the model sees.

Usage:
    python run_experiment27_multi_sample_robust_completion.py [--smoke]
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
# qmlreal_oracle_truth.csv is produced by this repository's own run_experiment5
# (the ambient oracle j*/gamma0 baseline) -- lives there, not duplicated here.
ORACLE_CSV = Path(__file__).resolve().parent / "data" / "qmlreal_oracle_truth.csv"
DATASET = "bloodmnist"
N_A = 4
SEEDS = [42, 43, 44, 45, 46]
STAGES = ["before", "after"]
N_SAMPLES = 5


def _load_ambient_oracle():
    with open(ORACLE_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["state_id"]: r for r in rows}


def run(smoke: bool):
    pool = build_pool(N_A)
    ambient = _load_ambient_oracle()
    out_path = RESULTS_DIR / "multi_sample_robust_completion.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    n_samples = 2 if smoke else N_SAMPLES

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

            rho_list, jac_list, r_dc_list, gdc_list = [], [], [], []
            for k in range(n_samples):
                ts_k = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=8, layer=8,
                                   ring_pairs=[(q, (q + 1) % 8) for q in range(8)], n_a=N_A,
                                   x_ref=x_ct[k:k + 1], final_ca=float("nan"), phis=phis)
                rho_k, jac_k = rho_A_and_jacobian(ts_k, stage)
                res_k = compute_R_manifold(rho_k, jac_k, pool)
                rho_list.append(rho_k); jac_list.append(jac_k)
                r_dc_list.append(res_k.r_DC_coef); gdc_list.append(res_k.gamma_D_C)

            r_dc_stack = np.stack(r_dc_list)                          # (n_samples, n_ops)
            off_scores = r_dc_stack[:, pool.offdiag_idx] ** 2         # (n_samples, n_offdiag)
            score_avg = off_scores.mean(axis=0)
            score_worst = off_scores.min(axis=0)
            j_robust_avg = int(np.argmax(score_avg))
            j_robust_worst = int(np.argmax(score_worst))

            j_ambient_local = int(ambient[state_id]["oracle_j_star"])
            candidates = {
                "ambient": j_ambient_local,
                "robust_avg": j_robust_avg,
                "robust_worst": j_robust_worst,
            }

            row = dict(state_id=state_id, seed=seed, stage=stage,
                       gamma_D_C_before_mean=float(np.mean(gdc_list)),
                       gamma_D_C_before_worst=float(np.max(gdc_list)),
                       j_ambient=pool.offdiag_labels[j_ambient_local],
                       j_robust_avg=pool.offdiag_labels[j_robust_avg],
                       j_robust_worst=pool.offdiag_labels[j_robust_worst],
                       robust_avg_eq_ambient=(j_robust_avg == j_ambient_local),
                       robust_worst_eq_ambient=(j_robust_worst == j_ambient_local),
                       robust_avg_eq_worst=(j_robust_avg == j_robust_worst))

            for cand_name, j_local in candidates.items():
                j_global = pool.offdiag_idx[j_local]
                after_vals = []
                for k in range(n_samples):
                    res_after = compute_R_manifold(rho_list[k], jac_list[k], pool,
                                                     extra_measured_idx=[j_global])
                    after_vals.append(res_after.gamma_D_C)
                row[f"gamma_D_C_after_{cand_name}_mean"] = float(np.mean(after_vals))
                row[f"gamma_D_C_after_{cand_name}_worst"] = float(np.max(after_vals))

            rows.append(row)
            elapsed = time.time() - t0
            print(f"[{len(rows)}/{len(seeds)*len(STAGES)}] {state_id}: "
                  f"before(mean/worst)={row['gamma_D_C_before_mean']:.3f}/{row['gamma_D_C_before_worst']:.3f} | "
                  f"ambient={row['j_ambient']} after(worst)={row['gamma_D_C_after_ambient_worst']:.3f} | "
                  f"robust_avg={row['j_robust_avg']} after(worst)={row['gamma_D_C_after_robust_avg_worst']:.3f} | "
                  f"robust_worst={row['j_robust_worst']} after(worst)={row['gamma_D_C_after_robust_worst_worst']:.3f} "
                  f"(elapsed {elapsed:.0f}s)", flush=True)
            _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path} ({time.time()-t0:.0f}s total).")
    _summarize(rows)
    return rows


def _summarize(rows):
    if not rows:
        return
    for cand in ["ambient", "robust_avg", "robust_worst"]:
        vals = [r[f"gamma_D_C_after_{cand}_worst"] for r in rows]
        print(f"  worst-case-remaining gamma_D_C after adding j*_{cand}: "
              f"mean={np.mean(vals):.4f} median={np.median(vals):.4f}")
    n_avg_eq_amb = sum(1 for r in rows if r["robust_avg_eq_ambient"])
    n_worst_eq_amb = sum(1 for r in rows if r["robust_worst_eq_ambient"])
    print(f"\n  j*_robust_avg == j*_ambient: {n_avg_eq_amb}/{len(rows)}")
    print(f"  j*_robust_worst == j*_ambient: {n_worst_eq_amb}/{len(rows)}")


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
