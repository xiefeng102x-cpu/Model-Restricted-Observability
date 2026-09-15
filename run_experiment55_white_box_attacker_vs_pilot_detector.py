"""Experiment 55 (README Follow-up 82; Phase-2 peer-review roadmap item,
responding to the Devil's Advocate's C7): Experiment 45's white-box
attacker optimizes against the TRUE, noise-free geometry drift and is
evaluated only via completion-recovery metrics (static recovery,
whether j* flips) -- never plugged into the actual pilot-based, noisy
end-to-end AUC detector (Experiment 38/54) a real deployment would use.
Does an attacker who drives the TRUE drift to ~0 (Experiment 45's own
finding) also evade the PILOT-BASED detector, or does pilot noise still
give the detector a signal the oracle-level completion metric misses?

Design: reproduces Experiment 45's exact optimization (same objective,
same random-search-then-Powell-refinement procedure, same derive_seed
draws, same EXPERIMENT_ID string so the RNG stream is bit-identical) to
recover the actual optimized phis per seed -- Experiment 45 persisted
only summary statistics, not the optimized phis themselves. The
recovered optimized_drift is printed against Experiment 45's own
already-published per-seed numbers as a sanity check that the
reproduction is faithful before trusting anything downstream. Builds
the full post-attack state from those phis and evaluates it with the
SAME pilot-based manifold-restricted score and calibration/test-split
protocol as Experiment 38/54, alongside the ORIGINAL random-direction
attacker at the identical magnitude for a paired before/after-
optimization comparison.

Usage:
    python run_experiment55_white_box_attacker_vs_pilot_detector.py [--smoke]
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
EXPERIMENT_ID = "paper13_exp55_whitebox_vs_pilot_detector_v1"
ATTACKER_EXPERIMENT_ID = "paper13_exp45_white_box_attacker_v1"  # must match Experiment 45 exactly to reproduce its RNG stream

# Experiment 45's own persisted optimized_drift, for the sanity check below.
EXP45_OPTIMIZED_DRIFT = {42: 1.9115980595095964e-14, 43: 1.0131328352382081e-14}


def _make_objective(flat_base, ring_pairs, jac_before, pool, r_dc_before_offdiag, target_norm, dim_a, dim_b):
    def objective(raw_phis):
        norm = np.linalg.norm(raw_phis)
        if norm < 1e-8:
            candidate_phis = np.zeros_like(raw_phis)
            candidate_phis[0] = target_norm
        else:
            candidate_phis = raw_phis / norm * target_norm
        with torch.no_grad():
            flat2 = mni.apply_zz_null_unitary(flat_base.unsqueeze(0), N_QUBITS,
                                               list(candidate_phis), ring_pairs).reshape(-1)
            M = flat2.reshape(dim_a, dim_b)
            rho_candidate = (M @ M.conj().T).numpy()
        res = compute_R_manifold(rho_candidate, jac_before, pool)
        r_dc_candidate = res.r_DC_coef[pool.offdiag_idx]
        return float(np.linalg.norm(r_dc_candidate - r_dc_before_offdiag))
    return objective


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


def _recover_optimized_phis(seed, pool):
    """Reproduces Experiment 45's exact optimization to recover the
    actual optimized phis, which that experiment did not persist."""
    art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
    theta0 = torch.tensor(art_before["theta"], dtype=torch.float64, requires_grad=False)
    rho_before, jac_before = art_before["rho_A"], art_before["jac"]
    cfg = di.DATASETS[DATASET]
    x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
    ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
    dim_a, dim_b = 2 ** N_A, 2 ** (N_QUBITS - N_A)

    with torch.no_grad():
        state = tms.simulate_final_state(x_ct[:1], theta0, N_QUBITS, LAYER, ring_pairs)
        flat_base = state.reshape(2 ** N_QUBITS)

    res_before = compute_R_manifold(rho_before, jac_before, pool)
    r_dc_before_offdiag = res_before.r_DC_coef[pool.offdiag_idx]

    rng_phi = np.random.RandomState(seed)
    original_phis = np.array(rng_phi.uniform(-1.5, 1.5, size=N_QUBITS))
    target_norm = float(np.linalg.norm(original_phis))

    objective = _make_objective(flat_base, ring_pairs, jac_before, pool, r_dc_before_offdiag,
                                 target_norm, dim_a, dim_b)

    rng_search = np.random.default_rng(
        derive_seed(ATTACKER_EXPERIMENT_ID, f"{DATASET}_seed{seed}", "search", "attacker"))
    best_val, best_raw = np.inf, None
    for _ in range(N_RANDOM_SEARCH):
        raw = rng_search.normal(size=N_QUBITS)
        val = objective(raw)
        if val < best_val:
            best_val, best_raw = val, raw

    opt = minimize(objective, best_raw, method="Powell", options={"maxiter": 300, "xtol": 1e-4})
    optimized_norm = np.linalg.norm(opt.x)
    optimized_phis = (opt.x / optimized_norm * target_norm) if optimized_norm > 1e-8 else original_phis

    return dict(rho_before=rho_before, jac_before=jac_before, r_dc_before_offdiag=r_dc_before_offdiag,
                flat_base=flat_base, ring_pairs=ring_pairs, dim_a=dim_a, dim_b=dim_b,
                original_phis=original_phis, optimized_phis=optimized_phis,
                optimized_drift=float(opt.fun))


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    n_calib = 4 if smoke else N_CALIB
    n_test_clean = 4 if smoke else N_TEST_CLEAN
    n_test_tampered = 4 if smoke else N_TEST_TAMPERED

    score_rows = []
    for seed in seeds:
        rec = _recover_optimized_phis(seed, pool)
        jac_before = rec["jac_before"]
        r_dc_before_offdiag = rec["r_dc_before_offdiag"]
        rho_before = rec["rho_before"]

        expected = EXP45_OPTIMIZED_DRIFT.get(seed)
        check = "" if expected is None else f" (Experiment 45's own persisted value: {expected:.6e})"
        print(f"seed={seed}: recovered optimized_drift={rec['optimized_drift']:.6e}{check}", flush=True)

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
            score_rows.append(dict(seed=seed, condition="white_box_optimized", label=1, score=sc))

        print(f"  seed={seed} done", flush=True)
        _write_csv(RESULTS_DIR / "white_box_attacker_vs_pilot_detector.csv", score_rows)

    print(f"\nDone collecting {len(score_rows)} scored trials.")
    _summarize(score_rows)


def _summarize(rows):
    labels_all = np.array([r["label"] for r in rows])
    scores_all = np.array([r["score"] for r in rows])
    clean_scores = scores_all[labels_all == 0]

    print("\n=== Per-condition AUC (each attacker condition vs. pooled clean) ===")
    for cond in ["random_direction", "white_box_optimized"]:
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
