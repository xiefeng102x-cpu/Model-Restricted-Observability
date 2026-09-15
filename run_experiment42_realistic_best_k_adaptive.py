"""Experiment 42 (README Follow-up 53; "item E" -- turning Ablation B's
characterized ceiling into an actual validated method, the way Experiment
37 turned the static/adaptive gap into the Gated Adaptive Audit).

Ablation B (Experiment 40) showed, using the TRUE noise-free oracle
gradient, that best-k Pauli completions recover far more of the ideal
continuous ceiling than a single Pauli (3.7% at k=1 vs 36.6% at k=20).
But that used the oracle g_after -- an unrealistic upper bound no real
defender has. This experiment asks the practical question directly: does
best-k selection ALSO help in the REALISTIC (pilot-estimated gradient +
trusted J_before) setting Follow-up 44/Ablation A actually uses? This is
a straightforward combination of already-validated pieces (Experiment
39's pilot-simulation call, Experiment 40's top-k ranking-and-evaluation
call) -- no new machinery, just the natural cross-product not yet tested.

Selection: rank offdiag candidates by |r_DC_coef|^2 using the PILOT-
ESTIMATED g_hat (via trusted J_before, exactly Follow-up 44's own
convention), take the top k, measure all of them
(`extra_measured_idx=top_k`). Evaluation: on the TRUE held-out after-
state, exactly as everywhere else in this project. Swept across k in
{1,2,3,5,10,20} and pilot budget in {10K,100K,1M}, on the same 5
BloodMNIST seeds and persisted artifacts Ablation A/B both used.

Recovery is normalized against the CONTINUOUS-ideal ceiling (Ablation
B's convention: 1 - gamma_D_C(after)/gamma_D_C(before)), NOT Ablation
A's single-Pauli-oracle convention -- deliberately, so this experiment's
k=1 realistic numbers are directly comparable on the same [0,100%] scale
as Ablation B's own k=1..20 ORACLE numbers, letting the two be read
side by side as "realistic vs. oracle at the same k." (Ablation A's own
normalization would let k>1 recovery exceed 100%, since multiple
realistic Paulis can out-perform the single-Pauli oracle used as its
denominator -- true, but a confusing way to report this experiment.)

Usage:
    python run_experiment42_realistic_best_k_adaptive.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.restricted_observability import compute_R_manifold, hermitian_log, pauli_coefficients
from measurement_simulator.pauli_pilot import simulate_pilot
from selector.reduced_state_plugin import select as route_a_select
from seeds import derive_seed

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
PILOT_BUDGETS = [10_000, 100_000, 1_000_000]
K_VALUES = [1, 2, 3, 5, 10, 20]
N_REPEATS = 20
EXPERIMENT_ID = "paper13_exp42_realistic_best_k_v1"


def run(smoke: bool):
    pool = build_pool(N_A)
    out_path = RESULTS_DIR / "realistic_best_k_adaptive.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    budgets = PILOT_BUDGETS[:1] if smoke else PILOT_BUDGETS
    k_values = [1, 5] if smoke else K_VALUES
    n_repeats = 3 if smoke else N_REPEATS

    rows = []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        art_after = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_after.npz")
        jac_before = art_before["jac"]
        rho_after, jac_after = art_after["rho_A"], art_after["jac"]

        baseline_after = compute_R_manifold(rho_after, jac_after, pool).gamma_D_C

        for n_pilot in budgets:
            recovery_by_k = {k: [] for k in k_values}
            for rep in range(n_repeats):
                s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"{n_pilot}_{rep}", "pilot")
                rng = np.random.default_rng(s)
                pilot = simulate_pilot(rho_after, pool.all_matrices, n_pilot, rng)
                route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
                g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)

                sel_res = compute_R_manifold(rho_after, jac_before, pool, g_coef_override=g_hat_coef)
                offdiag = np.array(pool.offdiag_idx)
                ranked = offdiag[np.argsort(-sel_res.exact_score[offdiag])]

                for k in k_values:
                    idx_k = list(ranked[:k])
                    eval_after = compute_R_manifold(rho_after, jac_after, pool,
                                                      extra_measured_idx=idx_k).gamma_D_C
                    recovery = 1.0 - eval_after / baseline_after
                    recovery_by_k[k].append(recovery)

            for k in k_values:
                mean_recovery = float(np.mean(recovery_by_k[k]))
                rows.append(dict(seed=seed, n_pilot=n_pilot, k=k, recovery=mean_recovery))
            k_max = max(k_values)
            print(f"seed={seed} n_pilot={n_pilot:>8}: k=1={np.mean(recovery_by_k[1])*100:5.1f}% "
                  f"k={k_max}={np.mean(recovery_by_k[k_max])*100:5.1f}%", flush=True)
            _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}.")
    _summarize(rows, budgets, k_values)


def _summarize(rows, budgets, k_values):
    print("\n=== Realistic best-k adaptive recovery: mean across seeds ===")
    for n_pilot in budgets:
        line = f"  n_pilot={n_pilot:>8}: "
        for k in k_values:
            sub = [r["recovery"] for r in rows if r["n_pilot"] == n_pilot and r["k"] == k]
            line += f"k={k}={np.mean(sub)*100:5.1f}%  "
        print(line)


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
