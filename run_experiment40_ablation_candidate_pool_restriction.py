"""Experiment 40 (README Follow-up 51; guide Section 15's "Ablation B —
candidate-pool restriction cost"). Orthogonal to Ablation A (Experiment 39,
which held the measurement BASIS fixed at "one real Pauli string" and
decomposed pilot-noise vs. geometry-mismatch cost): this ablation holds
the gradient/geometry source fixed at the TRUE oracle (`g_after`,
`J_after` -- no pilot noise, no `J_before` staleness) and instead varies
HOW MANY / WHAT KIND of measurement the completion is allowed to use,
answering the guide's own framing: "能告诉 reviewer：瓶颈来自 geometry、
estimation，还是 hardware basis restriction" (Ablation A already answers
geometry vs. estimation; this answers hardware-basis restriction).

Four conditions, all selected from the SAME true r_DC ranking and
evaluated via compute_R_manifold's extra_measured_idx / (new)
extra_measured_vectors on the true (rho_after, jac_after):

  1. ideal continuous B*=r_DC/||r_DC||  -- the hardware-unconstrained
     ceiling; Theorem B guarantees this drives gamma_D_C to exactly 0
     (recovery=100% by construction; validated against the module's own
     self-test check 5c/7 before use here).
  2. best single Pauli (k=1)            -- what every other experiment in
     this project has used as "the" completion.
  3. best-k Pauli linear combination, k in {1,2,3,5,10,20} -- greedy
     top-|r_DC_coef|^2 selection (the same ranking rule j* itself uses,
     extended to a set).
  4. hardware-compatible commuting group -- the maximal qubit-wise-
     commuting (QWC) set of offdiag candidates sharing ONE per-qubit
     measurement setting with the winning single Pauli (its own non-
     identity qubits fixed, identity positions padded with Z) -- i.e.
     "how much do you get for free from the SAME circuit execution that
     measures the best single Pauli."

Recovery is normalized as 1 - gamma_D_C(after completion) / gamma_D_C
(no completion), NOT Ablation A's own normalization (which used the
best-single-Pauli oracle as its ceiling) -- here the true ceiling is the
continuous direction, which reaches exactly 0, so this normalization is
the more natural one for THIS ablation. Stated explicitly to avoid
confusing the two ablations' numbers.

Zero retraining cost: reuses Experiment 25's persisted (rho_A, jac)
"after" artifacts unchanged.

Usage:
    python run_experiment40_ablation_candidate_pool_restriction.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.restricted_observability import compute_R_manifold

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
K_VALUES = [1, 2, 3, 5, 10, 20]


def _qwc_group_containing(j_star_global: int, pool) -> list:
    """Maximal qubit-wise-commuting offdiag group sharing j_star's own
    per-qubit measurement setting (its non-identity positions fixed;
    identity positions padded with Z=3, the "not actively used" default
    since Z is what the diagonal/already-measured basis uses)."""
    t_star = pool.all_tuples[j_star_global]
    setting = tuple(3 if x == 0 else x for x in t_star)
    group = []
    for i in pool.offdiag_idx:
        t = pool.all_tuples[i]
        if all(x == 0 or x == setting[q] for q, x in enumerate(t)):
            group.append(i)
    assert j_star_global in group
    return group


def run(smoke: bool):
    pool = build_pool(N_A)
    out_path = RESULTS_DIR / "ablation_candidate_pool_restriction.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    k_values = [1, 5] if smoke else K_VALUES

    rows = []
    for seed in seeds:
        art_after = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_after.npz")
        rho_after, jac_after = art_after["rho_A"], art_after["jac"]

        res_true = compute_R_manifold(rho_after, jac_after, pool)
        baseline = res_true.gamma_D_C
        offdiag = np.array(pool.offdiag_idx)
        ranked = offdiag[np.argsort(-res_true.exact_score[offdiag])]
        j_star = int(ranked[0])

        # 1. ideal continuous direction
        b_star_coef = res_true.r_DC_coef / np.linalg.norm(res_true.r_DC_coef)
        gamma_ideal = compute_R_manifold(rho_after, jac_after, pool,
                                          extra_measured_vectors=[b_star_coef]).gamma_D_C
        recovery_ideal = 1.0 - gamma_ideal / baseline
        rows.append(dict(seed=seed, condition="ideal_continuous", size=None,
                          gamma_after=gamma_ideal, recovery=recovery_ideal))

        # 2/3. best-k Pauli combination (k=1 == "best single Pauli")
        recovery_by_k = {}
        for k in k_values:
            idx_k = list(ranked[:k])
            gamma_k = compute_R_manifold(rho_after, jac_after, pool,
                                          extra_measured_idx=idx_k).gamma_D_C
            label = "best_single_pauli" if k == 1 else f"best_{k}_pauli_combo"
            recovery_k = 1.0 - gamma_k / baseline
            recovery_by_k[k] = recovery_k
            rows.append(dict(seed=seed, condition=label, size=k,
                              gamma_after=gamma_k, recovery=recovery_k))

        # 4. hardware-compatible QWC group containing j*
        group = _qwc_group_containing(j_star, pool)
        gamma_qwc = compute_R_manifold(rho_after, jac_after, pool,
                                        extra_measured_idx=group).gamma_D_C
        recovery_qwc = 1.0 - gamma_qwc / baseline
        rows.append(dict(seed=seed, condition="hardware_qwc_group", size=len(group),
                          gamma_after=gamma_qwc, recovery=recovery_qwc))

        k_max = max(k_values)
        print(f"seed={seed}: baseline={baseline:.4f} ideal={recovery_ideal*100:5.1f}% "
              f"single(k=1)={recovery_by_k[1]*100:5.1f}% "
              f"best_k={k_max}={recovery_by_k[k_max]*100:5.1f}% "
              f"qwc(size={len(group)})={recovery_qwc*100:5.1f}%", flush=True)
        _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}.")
    _summarize(rows, k_values)


def _summarize(rows, k_values):
    print("\n=== Ablation B summary (mean recovery across seeds) ===")
    conditions = ["ideal_continuous"] + \
        (["best_single_pauli"] if 1 in k_values else []) + \
        [f"best_{k}_pauli_combo" for k in k_values if k != 1] + \
        ["hardware_qwc_group"]
    for cond in conditions:
        sub = [r["recovery"] for r in rows if r["condition"] == cond]
        if not sub:
            continue
        sizes = [r["size"] for r in rows if r["condition"] == cond and r["size"] is not None]
        size_str = f" (size={np.mean(sizes):.1f})" if sizes else ""
        print(f"  {cond:>22}{size_str}: mean recovery = {np.mean(sub)*100:5.1f}%")


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
