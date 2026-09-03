"""Experiment 34 (README Follow-up 45): does Follow-up 42/43/44's
before/after-evasion-then-adaptive-fix pattern hold for MNIST too, not
just BloodMNIST?

MNIST's own R_manifold=1.0000 exactly (Follow-up 34, all 10/10 states,
zero exceptions) -- T_rho C is the FULL ambient space always for this
architecture (n_a=2, ambient dim=15 << 120 params, severely over-
parametrized), so gamma_D_C=gamma_D and r_DC equals g's offdiag part
EXACTLY. This means NO retraining and NO fresh Jacobian are needed here:
rho_A for all 10 MNIST states is already persisted by this repository (`states/
qml_reduced_states.py`, unchanged since Follow-up 5) -- a genuine,
free re-use, not a new computation.

To reuse (not reimplement) the exact same audited `compute_R_manifold`
code path as the BloodMNIST analysis, this script feeds it a trivial
"full-Pauli-basis-as-jac" -- `pool.all_matrices` reshaped to look like a
Jacobian whose "parameters" are the full nontrivial Pauli basis itself.
This makes compute_R_manifold's internal tangent-space rank trivially
equal the full ambient dimension by construction, exactly matching
MNIST's own already-established real behavior, without a second,
parallel "ambient-only" implementation that could silently drift from
the one this whole project's other numbers are built on.

Three checks, mirroring Follow-up 42/43/44 exactly:
  (a) does j*(before) == j*(after)? (already in results/qmlreal_oracle_
      truth.csv from a MUCH earlier experiment -- read directly, zero
      new computation)
  (b) cos(g_before_offdiag, g_after_offdiag) -- the ambient equivalent
      of Follow-up 43's r_DC cosine check
  (c) clean-calibrated vs. pilot-adaptive completion evasion test,
      reusing this repository's own Route A pilot/plug-in machinery unchanged
      (same as Follow-up 44) -- except here "adaptive" reduces to
      exactly this repository's original ambient plug-in selector, since there
      is no tangent-geometry mismatch possible when T_rho C is already
      the full ambient space for both before and after.

Usage:
    python run_experiment34_mnist_consistency_check.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from states.qml_reduced_states import load_qml_states
from measurement_simulator.pauli_pilot import simulate_pilot
from selector.reduced_state_plugin import select as route_a_select
from seeds import derive_seed
from manifold.restricted_observability import compute_R_manifold, hermitian_log, pauli_coefficients

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ORACLE_CSV = Path(__file__).resolve().parent / "data" / "qmlreal_oracle_truth.csv"
DATASET = "mnist"
N_A = 2
SEEDS = [42, 43, 44, 45, 46]
PILOT_BUDGETS = [10_000, 100_000, 1_000_000, 10_000_000]
N_REPEATS = 20
EXPERIMENT_ID = "paper13_exp34_mnist_consistency_v1"


def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def run(smoke: bool):
    pool = build_pool(N_A)
    n_ops = pool.all_matrices.shape[0]
    full_basis_jac = np.moveaxis(pool.all_matrices, 0, -1)   # (d,d,n_ops) -- trivial full-rank "jac"

    states = {s.state_id: s for s in load_qml_states() if s.dataset == DATASET}
    with open(ORACLE_CSV, newline="", encoding="utf-8") as f:
        ambient = {r["state_id"]: r for r in csv.DictReader(f) if r["dataset"] == DATASET}

    seeds = SEEDS[:1] if smoke else SEEDS
    budgets = PILOT_BUDGETS[:2] if smoke else PILOT_BUDGETS
    n_repeats = 3 if smoke else N_REPEATS

    summary_rows, adaptive_rows = [], []
    for seed in seeds:
        sid_before, sid_after = f"{DATASET}_seed{seed}_before", f"{DATASET}_seed{seed}_after"
        rho_before, rho_after = states[sid_before].rho, states[sid_after].rho

        res_before = compute_R_manifold(rho_before, full_basis_jac, pool)
        res_after = compute_R_manifold(rho_after, full_basis_jac, pool)
        cos_sim = _cos(res_before.r_DC_coef, res_after.r_DC_coef)

        j_before_local = int(ambient[sid_before]["oracle_j_star"])
        j_after_local = int(ambient[sid_after]["oracle_j_star"])
        same_j = j_before_local == j_after_local

        baseline_after = res_after.gamma_D  # == gamma_D_C here, R_manifold=1
        j_before_global = pool.offdiag_idx[j_before_local]
        clean_calibrated = compute_R_manifold(rho_after, full_basis_jac, pool,
                                                extra_measured_idx=[j_before_global]).gamma_D_C
        j_after_global = pool.offdiag_idx[j_after_local]
        oracle_after = compute_R_manifold(rho_after, full_basis_jac, pool,
                                           extra_measured_idx=[j_after_global]).gamma_D_C

        oracle_reduction = baseline_after - oracle_after
        clean_reduction = baseline_after - clean_calibrated
        clean_recovery = clean_reduction / oracle_reduction if oracle_reduction > 1e-9 else float("nan")

        summary_rows.append(dict(
            seed=seed, j_star_before=pool.offdiag_labels[j_before_local],
            j_star_after=pool.offdiag_labels[j_after_local], same_j_star=same_j,
            cos_r_DC_before_after=cos_sim, baseline_after=baseline_after,
            clean_calibrated=clean_calibrated, oracle_after=oracle_after,
            clean_recovery_fraction=clean_recovery,
        ))
        print(f"seed={seed}: j*_before={pool.offdiag_labels[j_before_local]} "
              f"j*_after={pool.offdiag_labels[j_after_local]} (same={same_j}) | "
              f"cos={cos_sim:.4f} | clean_recovery={clean_recovery*100:.1f}%", flush=True)

        for n_pilot in budgets:
            adaptive_vals = []
            for rep in range(n_repeats):
                pilot_seed = derive_seed(EXPERIMENT_ID, sid_before, f"{n_pilot}_{rep}", "pilot")
                rng = np.random.default_rng(pilot_seed)
                pilot = simulate_pilot(rho_after, pool.all_matrices, n_pilot, rng)
                route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
                g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)
                adaptive_res = compute_R_manifold(rho_after, full_basis_jac, pool, g_coef_override=g_hat_coef)
                j_adapt_local = int(np.argmax(adaptive_res.exact_score[pool.offdiag_idx]))
                j_adapt_global = pool.offdiag_idx[j_adapt_local]
                eval_res = compute_R_manifold(rho_after, full_basis_jac, pool, extra_measured_idx=[j_adapt_global])
                adaptive_vals.append(eval_res.gamma_D_C)
            adaptive_mean = float(np.mean(adaptive_vals))
            adaptive_reduction = baseline_after - adaptive_mean
            adaptive_recovery = adaptive_reduction / oracle_reduction if oracle_reduction > 1e-9 else float("nan")
            adaptive_rows.append(dict(seed=seed, n_pilot=n_pilot, n_repeats=n_repeats,
                                       clean_recovery_fraction=clean_recovery,
                                       adaptive_recovery_fraction_mean=adaptive_recovery))
            print(f"  n_pilot={n_pilot:>9}: adaptive recovery(mean over {n_repeats})={adaptive_recovery*100:.1f}%",
                  flush=True)

        _write_csv(RESULTS_DIR / "mnist_before_after_summary.csv", summary_rows)
        _write_csv(RESULTS_DIR / "mnist_adaptive_pilot_recalibration.csv", adaptive_rows)

    print(f"\nDone. {len(summary_rows)} summary rows, {len(adaptive_rows)} adaptive rows written.")
    _summarize(summary_rows, adaptive_rows)


def _summarize(summary_rows, adaptive_rows):
    n_same = sum(1 for r in summary_rows if r["same_j_star"])
    cos_vals = [r["cos_r_DC_before_after"] for r in summary_rows]
    clean_vals = [r["clean_recovery_fraction"] for r in summary_rows]
    print(f"\n  j*(before) == j*(after): {n_same}/{len(summary_rows)}")
    print(f"  cos(g_before,g_after): mean={np.mean(cos_vals):.4f} values={[round(c,3) for c in cos_vals]}")
    print(f"  clean-calibrated recovery: mean={np.mean(clean_vals)*100:.1f}%")
    by_budget = {}
    for r in adaptive_rows:
        by_budget.setdefault(r["n_pilot"], []).append(r["adaptive_recovery_fraction_mean"])
    for budget in sorted(by_budget):
        vals = by_budget[budget]
        print(f"  adaptive @ n_pilot={budget:>9}: mean={np.mean(vals)*100:.1f}%")


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
