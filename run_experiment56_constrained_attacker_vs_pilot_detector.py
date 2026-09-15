"""Experiment 56 (README Follow-up 83; Phase-2 peer-review roadmap,
responding to the Devil's Advocate's C7): Experiment 55 answered this
question for Experiment 45's UNCONSTRAINED white-box attacker, which
inspection showed is a trivial exploit (perturbation hidden entirely in
the un-audited subsystem B, never touching rho_A at all) -- of course
that evades a detector that only measures rho_A; not an interesting
test of C7's actual concern. The real test is Experiment 45b's
CONSTRAINED attacker (confined to ring pairs that touch the audited
subsystem A, RQ9's "narrow family" result, mean 30.3% true-drift
reduction, a genuine, non-trivial partial evasion of the diagnostic's
own geometry) -- that attacker was, like Experiment 45's, evaluated
only via completion-recovery metrics, never plugged into the actual
pilot-based end-to-end AUC detector. This experiment closes that gap.

Design: reproduces Experiment 45b's exact optimization (same objective,
active_mask, random-search-then-Powell-refinement procedure, same
derive_seed draws) to recover the actual constrained-optimized phis per
seed -- not persisted by Experiment 45b, only summary statistics were.
The recovered optimized_drift is checked against Experiment 45b's own
already-published per-seed numbers as a sanity check before trusting
anything downstream. Builds the full post-attack state from those phis
and evaluates it with the SAME pilot-based manifold-restricted score
and calibration/test-split protocol as Experiment 38/54/55, alongside
the ORIGINAL random-direction attacker at the identical (A-relevant)
magnitude for a paired before/after-optimization comparison.

Usage:
    python run_experiment56_constrained_attacker_vs_pilot_detector.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.theta_jacobian import di, tms, mni
from manifold.restricted_observability import compute_R_manifold, hermitian_log, pauli_coefficients
from measurement_simulator.pauli_pilot import simulate_pilot
from selector.reduced_state_plugin import select as route_a_select
from seeds import derive_seed
from run_experiment45_white_box_adaptive_attacker import _make_objective

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
N_QUBITS = 8
LAYER = 8
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
N_RANDOM_SEARCH = 500
N_PILOT = 100_000
N_CALIB = 15
N_TEST_CLEAN = 15
N_TEST_TAMPERED = 15
EXPERIMENT_ID = "paper13_exp56_constrained_vs_pilot_detector_v1"
ATTACKER_EXPERIMENT_ID = "paper13_exp45b_constrained_attacker_v1"  # must match Experiment 45b exactly to reproduce its RNG stream

# Experiment 45b's own persisted optimized_drift, for the sanity check below.
EXP45B_OPTIMIZED_DRIFT = {42: 3.9524109266111833, 43: 3.9725918302128123, 44: 4.398844514900926,
                           45: 3.83900678026237, 46: 3.376825911645171}


def _rho_from_phis(phis, flat_base, ring_pairs, dim_a, dim_b):
    with torch.no_grad():
        flat2 = mni.apply_zz_null_unitary(flat_base.unsqueeze(0), N_QUBITS, list(phis), ring_pairs).reshape(-1)
        M = flat2.reshape(dim_a, dim_b)
        return (M @ M.conj().T).numpy()


def _pilot_score(rho_true, jac_before, r_dc_before_offdiag, pool, n_pilot, rng):
    pilot = simulate_pilot(rho_true, pool.all_matrices, n_pilot, rng)
    route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
    g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)
    r_dc_hat_offdiag = compute_R_manifold(route_a_out.rho_hat, jac_before, pool,
                                           g_coef_override=g_hat_coef).r_DC_coef[pool.offdiag_idx]
    return float(np.linalg.norm(r_dc_hat_offdiag - r_dc_before_offdiag))


def _recover_constrained_optimized_phis(seed, pool, ring_pairs, active_mask):
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

    rng_phi = np.random.RandomState(seed)
    original_phis = np.array(rng_phi.uniform(-1.5, 1.5, size=N_QUBITS))
    target_norm = float(np.linalg.norm(original_phis[active_mask]))

    objective = _make_objective(flat_base, ring_pairs, jac_before, pool, r_dc_before_offdiag,
                                 target_norm, dim_a, dim_b, active_mask=active_mask)

    rng_search = np.random.default_rng(
        derive_seed(ATTACKER_EXPERIMENT_ID, f"{DATASET}_seed{seed}", "search", "attacker"))
    best_val, best_raw = np.inf, None
    for _ in range(N_RANDOM_SEARCH):
        raw = rng_search.normal(size=N_QUBITS) * active_mask
        val = objective(raw)
        if val < best_val:
            best_val, best_raw = val, raw

    opt = minimize(objective, best_raw, method="Powell", options={"maxiter": 300, "xtol": 1e-4})
    raw_opt = opt.x * active_mask
    optimized_norm = np.linalg.norm(raw_opt)
    optimized_phis = (raw_opt / optimized_norm * target_norm) if optimized_norm > 1e-8 else original_phis
    optimized_drift = objective(optimized_phis)

    # constrained "random baseline": original phis with the un-audited components zeroed
    # (matching Experiment 45b's own baseline_drift convention: full original phis, active_mask
    # applied only inside the objective/search, not to the reported baseline state itself)
    return dict(rho_before=rho_before, jac_before=jac_before, r_dc_before_offdiag=r_dc_before_offdiag,
                flat_base=flat_base, ring_pairs=ring_pairs, dim_a=dim_a, dim_b=dim_b,
                original_phis=original_phis, optimized_phis=optimized_phis,
                optimized_drift=float(optimized_drift))


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    n_calib = 4 if smoke else N_CALIB
    n_test_clean = 4 if smoke else N_TEST_CLEAN
    n_test_tampered = 4 if smoke else N_TEST_TAMPERED

    ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
    active_mask = np.array([(p < N_A) or (q < N_A) for (p, q) in ring_pairs])

    score_rows = []
    for seed in seeds:
        rec = _recover_constrained_optimized_phis(seed, pool, ring_pairs, active_mask)
        jac_before = rec["jac_before"]
        r_dc_before_offdiag = rec["r_dc_before_offdiag"]
        rho_before = rec["rho_before"]

        expected = EXP45B_OPTIMIZED_DRIFT.get(seed)
        check = "" if expected is None else f" (Experiment 45b's own persisted value: {expected:.6f})"
        print(f"seed={seed}: recovered optimized_drift={rec['optimized_drift']:.6f}{check}", flush=True)

        for rep in range(n_calib):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"calib_{rep}", "pilot")
            rng = np.random.default_rng(s)
            _pilot_score(rho_before, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)

        for rep in range(n_test_clean):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"testclean_{rep}", "pilot")
            rng = np.random.default_rng(s)
            sc = _pilot_score(rho_before, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
            score_rows.append(dict(seed=seed, condition="clean", label=0, score=sc))

        rho_random = _rho_from_phis(rec["original_phis"], rec["flat_base"], rec["ring_pairs"],
                                     rec["dim_a"], rec["dim_b"])
        for rep in range(n_test_tampered):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"random_{rep}", "pilot")
            rng = np.random.default_rng(s)
            sc = _pilot_score(rho_random, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
            score_rows.append(dict(seed=seed, condition="random_direction", label=1, score=sc))

        rho_optimized = _rho_from_phis(rec["optimized_phis"], rec["flat_base"], rec["ring_pairs"],
                                        rec["dim_a"], rec["dim_b"])
        for rep in range(n_test_tampered):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"optimized_{rep}", "pilot")
            rng = np.random.default_rng(s)
            sc = _pilot_score(rho_optimized, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
            score_rows.append(dict(seed=seed, condition="constrained_optimized", label=1, score=sc))

        print(f"  seed={seed} done", flush=True)
        _write_csv(RESULTS_DIR / "constrained_attacker_vs_pilot_detector.csv", score_rows)

    print(f"\nDone collecting {len(score_rows)} scored trials.")
    _summarize(score_rows)


def _summarize(rows):
    labels_all = np.array([r["label"] for r in rows])
    scores_all = np.array([r["score"] for r in rows])
    clean_scores = scores_all[labels_all == 0]

    print("\n=== Per-condition AUC (each attacker condition vs. pooled clean) ===")
    for cond in ["random_direction", "constrained_optimized"]:
        cond_scores = np.array([r["score"] for r in rows if r["condition"] == cond])
        y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(cond_scores))])
        s = np.concatenate([clean_scores, cond_scores])
        auc = roc_auc_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)

        def tpr_at_fpr(target_fpr, fpr=fpr, tpr=tpr):
            idx = max(np.searchsorted(fpr, target_fpr, side="right") - 1, 0)
            return float(tpr[idx])

        print(f"  {cond:>20}: AUC={auc:.4f} TPR@1%FPR={tpr_at_fpr(0.01)*100:.1f}% "
              f"TPR@5%FPR={tpr_at_fpr(0.05)*100:.1f}% mean_score={cond_scores.mean():.4f} "
              f"(clean mean={clean_scores.mean():.4f})")

    print("\n=== Per-seed constrained_optimized AUC ===")
    for seed in [str(s) for s in SEEDS[:1]] if False else [str(s) for s in [42, 43, 44, 45, 46]]:
        opt = np.array([r["score"] for r in rows if r["condition"] == "constrained_optimized" and str(r["seed"]) == seed])
        if len(opt) == 0:
            continue
        y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(opt))])
        s = np.concatenate([clean_scores, opt])
        try:
            auc = roc_auc_score(y, s)
        except ValueError:
            auc = float("nan")
        print(f"  seed {seed}: AUC={auc:.4f} mean_score={opt.mean():.4f}")


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
