"""Experiment 45b (README Follow-up 56, continued). Experiment 45's
unconstrained white-box attacker search found, in 4/5 seeds, a
near-EXACT-zero-drift solution -- but inspection showed the optimizer
had concentrated the entire perturbation on ring pair (4,5), which for
n_a=4 (audited qubits 0-3) lies ENTIRELY within the traced-out
subsystem B (qubits 4-7). By the partial-trace identity, a local
operation confined to B cannot affect rho_A at all (confirmed directly:
||rho_after_optimized - rho_before|| = 1.4e-16 for seed 42, i.e.
literally unchanged, not merely geometrically aligned) -- a TRIVIAL
evasion (hide outside the audited region), not a sophisticated attack
on the diagnostic's own detection geometry. This is a real, previously
unstated limitation of any reduced-subsystem audit, but it is a
different and much less interesting finding than "the r_DC-based
detector can be geometrically fooled."

This script asks the sharper, intended question: forced to use ONLY
ring pairs that touch the audited subsystem A (excluding pairs (4,5),
(5,6), (6,7), which are entirely within B for n_a=4 -- 5 of the
original 8 ring-pair angles remain: pairs (0,1),(1,2),(2,3),(3,4),(7,0)),
can an informed attacker still find a near-invisible direction? Same
objective, optimizer, and matched-magnitude convention as Experiment 45
(target norm now computed over the 5 A-relevant components of the
seed's own original phis, for a fair comparison restricted to the same
subspace).

Usage:
    python run_experiment45b_white_box_attacker_constrained.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.theta_jacobian import di, tms
from seeds import derive_seed
from run_experiment45_white_box_adaptive_attacker import (
    _make_objective, _static_recovery_for_phis, compute_R_manifold,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
N_QUBITS = 8
LAYER = 8
SEEDS = [42, 43, 44, 45, 46]
N_RANDOM_SEARCH = 500
EXPERIMENT_ID = "paper13_exp45b_constrained_attacker_v1"


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    n_random = 30 if smoke else N_RANDOM_SEARCH

    ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
    active_mask = np.array([(p < N_A) or (q < N_A) for (p, q) in ring_pairs])
    print(f"active (A-relevant) ring-pair indices: {np.flatnonzero(active_mask).tolist()} "
          f"of {N_QUBITS} total\n")

    rows = []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        theta0 = torch.tensor(art_before["theta"], dtype=torch.float64, requires_grad=False)
        rho_before, jac_before = art_before["rho_A"], art_before["jac"]
        cfg = di.DATASETS[DATASET]
        x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
        dim_a, dim_b = 2 ** N_A, 2 ** (N_QUBITS - N_A)

        with torch.no_grad():
            state = tms.simulate_final_state(x_ct[:1], theta0, N_QUBITS, LAYER, ring_pairs)
            flat_base = state.reshape(2 ** N_QUBITS)

        res_before = compute_R_manifold(rho_before, jac_before, pool)
        r_dc_before_offdiag = res_before.r_DC_coef[pool.offdiag_idx]
        exact_score_before_offdiag = res_before.exact_score[pool.offdiag_idx]
        j_before = pool.offdiag_idx[int(np.argmax(exact_score_before_offdiag))]

        rng_phi = np.random.RandomState(seed)
        original_phis = np.array(rng_phi.uniform(-1.5, 1.5, size=N_QUBITS))
        target_norm = float(np.linalg.norm(original_phis[active_mask]))

        objective = _make_objective(flat_base, ring_pairs, jac_before, pool, r_dc_before_offdiag,
                                     target_norm, dim_a, dim_b, active_mask=active_mask)

        baseline_drift = objective(original_phis)
        baseline_static, baseline_flip = _static_recovery_for_phis(
            original_phis, flat_base, ring_pairs, jac_before, jac_before, pool, j_before, dim_a, dim_b)

        rng_search = np.random.default_rng(derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", "search", "attacker"))
        best_val, best_raw = np.inf, None
        for _ in range(n_random):
            raw = rng_search.normal(size=N_QUBITS) * active_mask
            val = objective(raw)
            if val < best_val:
                best_val, best_raw = val, raw

        opt = minimize(objective, best_raw, method="Powell", options={"maxiter": 300, "xtol": 1e-4})
        raw_opt = opt.x * active_mask
        optimized_norm = np.linalg.norm(raw_opt)
        optimized_phis = (raw_opt / optimized_norm * target_norm) if optimized_norm > 1e-8 else original_phis
        optimized_drift = objective(optimized_phis)
        optimized_static, optimized_flip = _static_recovery_for_phis(
            optimized_phis, flat_base, ring_pairs, jac_before, jac_before, pool, j_before, dim_a, dim_b)

        row = dict(seed=seed, target_norm=target_norm,
                   baseline_drift=baseline_drift, optimized_drift=optimized_drift,
                   baseline_static_recovery=baseline_static, optimized_static_recovery=optimized_static,
                   baseline_j_flipped=baseline_flip, optimized_j_flipped=optimized_flip)
        rows.append(row)
        print(f"seed={seed}: baseline_drift={baseline_drift:.4f} optimized_drift={optimized_drift:.4f} "
              f"(reduction={100*(1-optimized_drift/baseline_drift):.1f}%) "
              f"baseline_static={baseline_static*100:.1f}% optimized_static={optimized_static*100:.1f}% "
              f"baseline_flip={baseline_flip} optimized_flip={optimized_flip}", flush=True)
        _write_csv(RESULTS_DIR / "white_box_attacker_constrained.csv", rows)

    print(f"\nDone. {len(rows)} rows written.")
    _summarize(rows)


def _summarize(rows):
    print("\n=== Constrained (A-relevant-only) white-box attacker summary ===")
    bd = np.mean([r["baseline_drift"] for r in rows])
    od = np.mean([r["optimized_drift"] for r in rows])
    bs = np.mean([r["baseline_static_recovery"] for r in rows])
    os_ = np.mean([r["optimized_static_recovery"] for r in rows])
    n_still_flip = sum(1 for r in rows if r["optimized_j_flipped"])
    print(f"  drift: baseline={bd:.4f} -> optimized={od:.4f} ({100*(1-od/bd):.1f}% reduction)")
    print(f"  static recovery: baseline={bs*100:.1f}% -> optimized={os_*100:.1f}%")
    print(f"  j* still flips under optimized attack: {n_still_flip}/{len(rows)} seeds")


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
