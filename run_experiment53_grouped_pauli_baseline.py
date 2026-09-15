"""Experiment 53 (README Follow-up 68; guide/model_restricted_observability_
theory_revision_plan.md Section 14.1: "至少增加一个 ... 比较对象: 1. naive
per-operator allocation; 2. grouped tomography; 3. classical shadows").

Extends Experiment 44's direct-vs-shadow pilot comparison to a three-way
comparison by adding `measurement_simulator_grouped.simulate_grouped_pilot`
(validated via its own self-test) as a third acquisition strategy, at the
SAME matched total shot budgets and feeding the IDENTICAL downstream
single-Pauli adaptive-recovery pipeline as Experiment 44 -- only the
measurement-acquisition step differs across the three columns.

Grouped tomography (qubit-wise-commuting Pauli grouping, the standard VQE
measurement-grouping technique, already used in this project's Ablation B
for a different purpose) sits, by construction, strictly between direct
tomography's 4^a-1=255 dedicated per-operator settings and classical
shadows' random-basis-per-shot approach: it uses 3^a=81 FIXED settings
(same count as shadows' basis alphabet), but -- unlike shadows -- commits
the ENTIRE budget for a setting to one basis, non-randomly, and needs no
variance-inflating shadow-inversion correction factor since every shot
within a setting genuinely used that exact basis throughout.

Usage:
    python run_experiment53_grouped_pauli_baseline.py [--smoke]
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
from measurement_simulator_shadow import simulate_shadow_pilot
from measurement_simulator_grouped import simulate_grouped_pilot
from selector.reduced_state_plugin import select as route_a_select
from seeds import derive_seed

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
BUDGETS = [100, 500, 2_000, 10_000, 100_000, 1_000_000]
N_REPEATS = 20
EXPERIMENT_ID = "paper13_exp53_grouped_pilot_v1"


def _recovery_from_chat(chat, rho_after, jac_before, jac_after, pool, baseline_after, oracle_reduction):
    route_a_out = route_a_select(chat, pool.all_matrices, pool.offdiag_idx, pool.d)
    g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)
    sel_res = compute_R_manifold(rho_after, jac_before, pool, g_coef_override=g_hat_coef)
    j_hat = pool.offdiag_idx[int(np.argmax(sel_res.exact_score[pool.offdiag_idx]))]
    eval_after = compute_R_manifold(rho_after, jac_after, pool, extra_measured_idx=[j_hat]).gamma_D_C
    return (baseline_after - eval_after) / oracle_reduction if oracle_reduction > 1e-9 else float("nan")


def run(smoke: bool):
    pool = build_pool(N_A)
    out_path = RESULTS_DIR / "grouped_pauli_baseline_comparison.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    budgets = [100, 10_000] if smoke else BUDGETS
    n_repeats = 3 if smoke else N_REPEATS

    rows = []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        art_after = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_after.npz")
        jac_before = art_before["jac"]
        rho_after, jac_after = art_after["rho_A"], art_after["jac"]

        baseline_after = compute_R_manifold(rho_after, jac_after, pool).gamma_D_C
        oracle_sel = compute_R_manifold(rho_after, jac_after, pool)
        j_oracle = pool.offdiag_idx[int(np.argmax(oracle_sel.exact_score[pool.offdiag_idx]))]
        oracle_after = compute_R_manifold(rho_after, jac_after, pool, extra_measured_idx=[j_oracle]).gamma_D_C
        oracle_reduction = baseline_after - oracle_after

        for n_pilot in budgets:
            direct_recoveries, shadow_recoveries, grouped_recoveries = [], [], []
            for rep in range(n_repeats):
                s_direct = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"{n_pilot}_{rep}", "direct")
                rng_direct = np.random.default_rng(s_direct)
                direct = simulate_pilot(rho_after, pool.all_matrices, n_pilot, rng_direct)
                direct_recoveries.append(_recovery_from_chat(
                    direct.chat, rho_after, jac_before, jac_after, pool, baseline_after, oracle_reduction))

                s_shadow = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"{n_pilot}_{rep}", "shadow")
                rng_shadow = np.random.default_rng(s_shadow)
                shadow = simulate_shadow_pilot(rho_after, pool, n_pilot, rng_shadow)
                shadow_recoveries.append(_recovery_from_chat(
                    shadow.chat, rho_after, jac_before, jac_after, pool, baseline_after, oracle_reduction))

                s_grouped = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"{n_pilot}_{rep}", "grouped")
                rng_grouped = np.random.default_rng(s_grouped)
                grouped = simulate_grouped_pilot(rho_after, pool, n_pilot, rng_grouped)
                grouped_recoveries.append(_recovery_from_chat(
                    grouped.chat, rho_after, jac_before, jac_after, pool, baseline_after, oracle_reduction))

            rows.append(dict(seed=seed, n_pilot=n_pilot,
                              direct_recovery=float(np.mean(direct_recoveries)),
                              shadow_recovery=float(np.mean(shadow_recoveries)),
                              grouped_recovery=float(np.mean(grouped_recoveries)),
                              shots_per_setting_direct=n_pilot // len(pool.all_matrices),
                              shots_per_setting_grouped=n_pilot // (3 ** N_A)))
            print(f"seed={seed} n_pilot={n_pilot:>8}: "
                  f"direct={np.mean(direct_recoveries)*100:5.1f}% "
                  f"shadow={np.mean(shadow_recoveries)*100:5.1f}% "
                  f"grouped={np.mean(grouped_recoveries)*100:5.1f}%",
                  flush=True)
            _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}.")
    _summarize(rows, budgets)


def _summarize(rows, budgets):
    print("\n=== Direct vs. shadow vs. grouped pilot: mean recovery across seeds ===")
    for n_pilot in budgets:
        sub = [r for r in rows if r["n_pilot"] == n_pilot]
        if not sub:
            continue
        d = np.mean([r["direct_recovery"] for r in sub])
        sh = np.mean([r["shadow_recovery"] for r in sub])
        g = np.mean([r["grouped_recovery"] for r in sub])
        print(f"  n_pilot={n_pilot:>8}: direct={d*100:5.1f}%  shadow={sh*100:5.1f}%  grouped={g*100:5.1f}%")


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
