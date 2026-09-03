"""Experiment 41 (README Follow-up 52). Follow-up 50 (Ablation A) found
the geometry-mismatch-only recovery ceiling `c` varies ~20x across the 5
BloodMNIST seeds (4.3%-77.0%) and the manuscript Discussion (item 8)
asserts a defender "cannot know in advance ... how much this design will
ultimately recover" -- an INFERENCE beyond what Ablation A directly
tested. Ablation A only showed the ceiling varies; it never checked
whether that variation correlates with anything computable from the
TRUSTED BEFORE-STATE ALONE (i.e., before any tampering/deployment, using
only information a real defender actually has in advance). If such a
correlate existed, the "cannot know in advance" claim would be false or
overstated.

This experiment checks 5 natural before-state-only candidate predictors
against the known per-seed ceiling `c`:

  1. gamma_D_before      -- ambient blind-gradient magnitude at the
                             trusted state (this repository's existing quantity)
  2. gamma_D_C_before    -- manifold-restricted blind-gradient magnitude
                             at the trusted state (= static baseline
                             recovery target throughout Follow-up 42-44)
  3. R_manifold_before   -- gamma_D_C_before / gamma_D_before, this
                             paper's own headline ratio
  4. tangent_rank_before -- dim(T_rho C) at the trusted state
  5. eff_rank_before     -- participation-ratio effective rank of
                             jac_before's singular-value spectrum,
                             (sum S)^2 / sum(S^2) -- a "how generic/
                             isotropic is the reachable tangent
                             geometry" measure, distinct from the raw
                             (integer) tangent_rank

With only n=5 seeds this cannot establish or rule out a real
relationship with any statistical power -- reported as an honest,
small-n correlation check, not a validated predictive model.

Usage:
    python run_experiment41_ceiling_predictability_check.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.restricted_observability import compute_R_manifold, pauli_coefficients

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
SEEDS = [42, 43, 44, 45, 46]


def _effective_rank(jac, pool) -> float:
    jac_moved = np.moveaxis(jac, -1, 0)
    jac_coef = pauli_coefficients(jac_moved, pool.all_matrices).T
    S = np.linalg.svd(jac_coef, full_matrices=False, compute_uv=False)
    S = S[S > 1e-8 * S.max()]
    return float((S.sum() ** 2) / (S ** 2).sum())


def run():
    pool = build_pool(N_A)

    # per-seed geometry-mismatch ceiling `c`, re-read directly from
    # Ablation A's own persisted CSV (not transcribed from memory/README)
    import csv as csv_mod
    ceiling = {}
    with open(RESULTS_DIR / "ablation_geometry_mismatch.csv") as f:
        for row in csv_mod.DictReader(f):
            ceiling[int(row["seed"])] = float(row["recovery_c_geometry_mismatch_only"])
    assert len(ceiling) == 5, f"expected 5 seeds in ablation_geometry_mismatch.csv, got {len(ceiling)}"

    records = []
    for seed in SEEDS:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        rho_before, jac_before = art_before["rho_A"], art_before["jac"]
        res = compute_R_manifold(rho_before, jac_before, pool)
        rec = dict(
            seed=seed,
            gamma_D_before=res.gamma_D,
            gamma_D_C_before=res.gamma_D_C,
            R_manifold_before=res.R_manifold,
            tangent_rank_before=res.tangent_rank,
            eff_rank_before=_effective_rank(jac_before, pool),
            ceiling_c=ceiling[seed],
        )
        records.append(rec)
        print(f"seed={seed}: gamma_D={rec['gamma_D_before']:.4f} "
              f"gamma_D_C={rec['gamma_D_C_before']:.4f} "
              f"R_manifold={rec['R_manifold_before']:.4f} "
              f"tangent_rank={rec['tangent_rank_before']} "
              f"eff_rank={rec['eff_rank_before']:.2f} "
              f"ceiling_c={rec['ceiling_c']*100:.1f}%")

    predictors = ["gamma_D_before", "gamma_D_C_before", "R_manifold_before",
                  "tangent_rank_before", "eff_rank_before"]
    c_vals = np.array([r["ceiling_c"] for r in records])

    print(f"\n=== Correlation with ceiling c (n={len(records)} seeds -- LOW POWER, informal only) ===")
    for p in predictors:
        x = np.array([r[p] for r in records], dtype=float)
        if np.allclose(x, x[0]):
            print(f"  {p:>20}: constant across seeds (no variation to correlate)")
            continue
        r_pear, p_pear = pearsonr(x, c_vals)
        r_spear, p_spear = spearmanr(x, c_vals)
        print(f"  {p:>20}: Pearson r={r_pear:+.3f} (p={p_pear:.3f})  "
              f"Spearman rho={r_spear:+.3f} (p={p_spear:.3f})")

    out_path = RESULTS_DIR / "ceiling_predictability_check.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nDone. {len(records)} rows written to {out_path}.")


if __name__ == "__main__":
    run()
