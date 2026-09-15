"""Experiment 26: is j*_C (Follow-up 35's manifold-aware measurement pick)
stable, or is it chasing local noise? Directly answers the open question
flagged at the end of Follow-up 35 -- before treating any single j*_C as
an operational recommendation, check whether it survives (a) a different
choice of reference sample x_ref, and (b) a small perturbation of theta
(a cheap proxy for "a nearby training run converged to a slightly
different point").

Uses the theta/rho_A persisted by Experiment 25 (results/manifold_
artifacts/*.npz) -- no retraining. Only the forward simulation + autograd
Jacobian is recomputed per variant (seconds, not the ~15-minute BloodMNIST
retrain).

Usage:
    python run_experiment26_j_star_C_stability.py [--smoke]
"""
import argparse
import csv
import dataclasses
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
N_XREF_VARIANTS = 4          # alternative same-class reference samples
THETA_EPS_RELS = [0.001, 0.01]
N_THETA_REPEATS = 5


def _j_star_C(rho_A, jac, pool):
    res = compute_R_manifold(rho_A, jac, pool)
    score = res.exact_score[pool.offdiag_idx]
    order = np.argsort(-score)
    j_star = int(order[0])
    margin = float(score[order[0]] - score[order[1]]) if len(order) > 1 else float("nan")
    return j_star, margin, res.R_manifold


def run(smoke: bool):
    pool = build_pool(N_A)
    out_path = RESULTS_DIR / "j_star_C_stability.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    n_xref = 2 if smoke else N_XREF_VARIANTS
    n_theta_rep = 2 if smoke else N_THETA_REPEATS

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
                # reproduce the exact seed-derived phis (deterministic, same recipe as theta_jacobian.py)
                rng_phi = np.random.RandomState(seed)
                phis = list(rng_phi.uniform(-1.5, 1.5, size=8))
            base_ts = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=8, layer=8,
                                  ring_pairs=[(q, (q + 1) % 8) for q in range(8)], n_a=N_A,
                                  x_ref=x_ct[:1], final_ca=float("nan"), phis=phis)

            rho0, jac0 = rho_A_and_jacobian(base_ts, stage)
            j0, m0, R0 = _j_star_C(rho0, jac0, pool)
            r0_manifold_stored = float(art["R_manifold"])
            # 1e-6 was tight enough for a single-process, single-thread-count
            # comparison; this script recomputes R0 independently from the
            # value experiment25 stored, and multi-threaded BLAS/SVD
            # reduction order is not guaranteed bit-identical across
            # separate process launches even at a fixed OMP_NUM_THREADS
            # (confirmed empirically: two separate reruns gave 8.8e-6 and
            # 2.45e-6 disagreement on the same state, both far below any
            # plausible real-bug magnitude). 1e-4 comfortably covers this
            # noise floor while still catching a genuinely wrong/stale
            # cached artifact (which would differ by orders of magnitude).
            assert abs(R0 - r0_manifold_stored) < 1e-4, (state_id, R0, r0_manifold_stored)

            n_xref_agree, n_xref_total = 0, 0
            for k in range(1, n_xref + 1):
                if k >= x_ct.shape[0]:
                    break
                ts_var = dataclasses.replace(base_ts, x_ref=x_ct[k:k + 1])
                rho_v, jac_v = rho_A_and_jacobian(ts_var, stage)
                jv, mv, Rv = _j_star_C(rho_v, jac_v, pool)
                n_xref_total += 1
                n_xref_agree += int(jv == j0)
                rows.append(dict(state_id=state_id, variant_type="x_ref", variant_id=k,
                                  j_star_C=jv, margin_C=mv, R_manifold=Rv,
                                  agrees_with_original=(jv == j0)))

            n_theta_agree, n_theta_total = 0, 0
            for eps_rel in THETA_EPS_RELS:
                rms = float(theta0.detach().norm()) / (theta0.numel() ** 0.5)
                for rep in range(n_theta_rep):
                    rng = np.random.RandomState(seed * 1000 + rep)
                    noise = rng.normal(scale=eps_rel * rms, size=theta0.numel())
                    theta_pert = (theta0.detach() + torch.tensor(noise, dtype=torch.float64)).requires_grad_(True)
                    ts_pert = dataclasses.replace(base_ts, theta=theta_pert)
                    rho_p, jac_p = rho_A_and_jacobian(ts_pert, stage)
                    jp, mp, Rp = _j_star_C(rho_p, jac_p, pool)
                    n_theta_total += 1
                    n_theta_agree += int(jp == j0)
                    rows.append(dict(state_id=state_id, variant_type=f"theta_eps{eps_rel}", variant_id=rep,
                                      j_star_C=jp, margin_C=mp, R_manifold=Rp,
                                      agrees_with_original=(jp == j0)))

            elapsed = time.time() - t0
            print(f"{state_id}: j0={pool.offdiag_labels[j0]} (margin={m0:.4f}) | "
                  f"x_ref stability {n_xref_agree}/{n_xref_total} | "
                  f"theta-perturbation stability {n_theta_agree}/{n_theta_total} "
                  f"(elapsed {elapsed:.0f}s)", flush=True)
            _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path} ({time.time()-t0:.0f}s total).")
    _summarize(rows)
    return rows


def _summarize(rows):
    for vtype in sorted(set(r["variant_type"] for r in rows)):
        sub = [r for r in rows if r["variant_type"] == vtype]
        n_agree = sum(1 for r in sub if r["agrees_with_original"])
        print(f"  {vtype}: {n_agree}/{len(sub)} agree with the original (unperturbed) j*_C")
    xref_rows = [r for r in rows if r["variant_type"] == "x_ref"]
    theta_rows = [r for r in rows if r["variant_type"].startswith("theta_eps")]
    if xref_rows:
        n = sum(1 for r in xref_rows if r["agrees_with_original"])
        print(f"\n  TOTAL x_ref stability: {n}/{len(xref_rows)}")
    if theta_rows:
        n = sum(1 for r in theta_rows if r["agrees_with_original"])
        print(f"  TOTAL theta-perturbation stability: {n}/{len(theta_rows)}")


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
