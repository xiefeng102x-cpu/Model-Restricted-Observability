"""Manifold-restricted local observability (this project's
next_stage_operational_observability_research_guide.md, Section 2.2-2.4;
README Follow-up 34).

Ambient gamma_D (this repository's existing quantity -- already computed for all
20 real QML states, see oracle/entropy_target.py's `gamma0` /
results/qmlreal_oracle_truth.csv's `gamma0_true` column) measures how much
of the diagnostic's HS-gradient lies outside the measured operator span
S_M, among ALL possible Hermitian perturbation directions of rho_A.

gamma_{D,C} restricts that question to only the perturbation directions
theta-parametrized QML training can actually reach at this specific
(x_ref, theta) point: T_rho C = image of d(rho_A)/d(theta), the model's
own tangent space here. R_manifold = gamma_{D,C} / gamma_D << 1 would mean
the model's own constraints make the state easier to audit than the
ambient (worst-case, arbitrary-density-matrix) analysis suggests -- the
new guide's headline candidate direction.

Implementation note (the one real subtlety here): Hermitian matrices form
a REAL vector subspace of C^{dxd} (i*H is generally not Hermitian even
when H is), so an ordinary COMPLEX SVD of vectorized Hermitian matrices
does not, in general, return a basis whose vectors are themselves
Hermitian. Everything below instead works in REAL Pauli-coefficient
coordinates (coef_m(H) = Tr[P_m @ H]/sqrt(d) for Hermitian H -- real,
because Tr of a product of two Hermitian matrices is always real: for
Hermitian A,B, (AB)^dagger = BA, so Tr[AB] = Tr[(AB)^dagger] = Tr[BA] =
Tr[AB] by cyclicity, i.e. Tr[AB] equals its own complex conjugate). This
is exactly oracle/entropy_target.py's own a_j convention, just applied to
the tangent-space basis vectors too, and sidesteps the complex-SVD trap
entirely.
"""
from dataclasses import dataclass

import numpy as np


def hermitian_log(rho: np.ndarray, eig_floor: float = 1e-14) -> np.ndarray:
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, eig_floor, None)
    return (v * np.log(w)) @ v.conj().T


def pauli_coefficients(mats: np.ndarray, all_matrices: np.ndarray) -> np.ndarray:
    """mats: (..., d, d) complex Hermitian. all_matrices: (n_ops, d, d), the
    full nontrivial Pauli basis (candidates.pauli_pool.PauliPool.all_matrices).
    Returns real (..., n_ops), coef_m = Tr[P_m @ H] / sqrt(d)."""
    d = all_matrices.shape[-1]
    coef = np.einsum("mij,...ji->...m", all_matrices, mats) / np.sqrt(d)
    return coef.real


@dataclass
class ManifoldObservabilityResult:
    gamma_D: float           # ambient blind-gradient amplitude
    gamma_D_C: float         # manifold-restricted blind-gradient amplitude
    R_manifold: float        # gamma_D_C / gamma_D
    tangent_rank: int        # dim(T_rho C) <= n_params
    null_dim: int             # dim(N_M) within T_rho C
    g_coef: np.ndarray        # (n_ops,) real -- ambient ubar ambient HS-gradient's Pauli coefficients
    r_DC_coef: np.ndarray     # (n_ops,) real -- guide Theorem B's completion direction r_{D,C} =
                               # Pi_{N_M} g_{D,C}, expressed back in ambient Pauli-coefficient
                               # coordinates (same convention as oracle.entropy_target's a_j, so
                               # r_DC_coef[offdiag_idx] is directly comparable/combinable with it).
                               # ||r_DC_coef|| == gamma_D_C by construction.
    exact_score: np.ndarray   # (n_ops,) real -- Section 4's EXACT one-candidate reduction score
                               # gamma_j^2's reduction term |<r_DC,P_j>|^2 / ||P_N P_j||^2 (README
                               # Follow-up 96; run_experiment52's ||q_j|| computation, generalized
                               # and folded into this function so every caller gets it for free).
                               # argmax_j exact_score[j] over pool.offdiag_idx is the theoretically
                               # optimal single-candidate pick; argmax_j r_DC_coef[j]**2 is the
                               # older numerator-only heuristic these two agree on 8/10 real
                               # BloodMNIST states (Experiment 52) and disagree by 3-5% relative
                               # sub-optimality on the other 2. Candidates with ||P_N P_j||~0 (not
                               # reachable within the current null space at all) get score 0, not
                               # inf/nan.


def compute_R_manifold(rho_A: np.ndarray, jac: np.ndarray, pool,
                        tangent_tol: float = 1e-8, null_tol: float = 1e-6,
                        extra_measured_idx=None, g_coef_override=None,
                        extra_measured_vectors=None) -> ManifoldObservabilityResult:
    """jac: (d,d,n_params) complex128, d(rho_A)/d(theta_i) per column
    (manifold.theta_jacobian.rho_A_and_jacobian's output). pool: a
    candidates.pauli_pool.PauliPool built for this state's `a`.

    extra_measured_idx: optional list of GLOBAL pool indices (e.g. one
    entry from pool.offdiag_idx) to add to the measured set on top of
    pool.diag_idx -- guide Theorem B/C's "what if we add this one new
    observable" completion check, without needing a second module.

    g_coef_override: optional (n_ops,) real Pauli-coefficient vector to
    use IN PLACE OF -log(rho_A)'s own coefficients -- lets the tangent
    geometry (jac, i.e. T_rho C) come from one state (e.g. a defender's
    own KNOWN/trusted "before" configuration) while the diagnostic
    gradient comes from a DIFFERENT source (e.g. a pilot-measurement
    plug-in estimate of the actual deployed, possibly-perturbed state) --
    the adaptive-recalibration use case (README Follow-up 44).

    extra_measured_vectors: optional list of (n_ops,) real ambient
    Pauli-coefficient vectors (need not be a single pool basis element,
    e.g. r_DC/||r_DC|| itself) to add as extra measured ROWS alongside
    extra_measured_idx -- generalizes extra_measured_idx from "measure
    one more real hardware-realizable Pauli string" to "measure one more
    arbitrary generalized observable," used by Ablation B (README
    Follow-up 50) to quantify the idealized, hardware-unconstrained
    completion ceiling (self_test's check 5c does this same computation
    by hand; this parameter is the reusable, tested version of it)."""
    g_coef = g_coef_override if g_coef_override is not None else pauli_coefficients(-hermitian_log(rho_A), pool.all_matrices)

    gamma_D = float(np.linalg.norm(g_coef[pool.offdiag_idx]))
    n_ops = g_coef.shape[0]

    jac_moved = np.moveaxis(jac, -1, 0)                                    # (n_params,d,d)
    jac_coef = pauli_coefficients(jac_moved, pool.all_matrices).T          # (n_ops, n_params) real

    U, S, _ = np.linalg.svd(jac_coef, full_matrices=False)
    smax = S.max() if S.size else 0.0
    rank = int((S > tangent_tol * max(smax, 1e-300)).sum())

    if rank == 0:
        return ManifoldObservabilityResult(gamma_D=gamma_D, gamma_D_C=0.0, R_manifold=0.0,
                                            tangent_rank=0, null_dim=0, g_coef=g_coef,
                                            r_DC_coef=np.zeros(n_ops), exact_score=np.zeros(n_ops))

    E = U[:, :rank]                                                        # (n_ops, rank) real orthonormal

    measured_idx = list(pool.diag_idx) + list(extra_measured_idx or [])
    A = E[measured_idx, :]                                                 # (n_measured, rank)
    if extra_measured_vectors:
        extra_rows = np.stack([np.asarray(vec) @ E for vec in extra_measured_vectors], axis=0)
        A = np.vstack([A, extra_rows])                                     # (n_measured+n_extra, rank)
    _, Sa, Vha = np.linalg.svd(A, full_matrices=True)
    smax_a = Sa.max() if Sa.size else 0.0
    rank_a = int((Sa > null_tol * max(smax_a, 1e-300)).sum())
    N = Vha.T[:, rank_a:]                                                  # (rank, null_dim) real orthonormal
    null_dim = N.shape[1]

    c_proj = E.T @ g_coef                                                  # (rank,) real, g_{D,C} in T_rho C coords
    if null_dim > 0:
        n_coeffs = N.T @ c_proj                                            # (null_dim,) coords within N_M
        r_DC_in_tangent = N @ n_coeffs                                     # (rank,) back in T_rho C coords
        r_DC_coef = E @ r_DC_in_tangent                                    # (n_ops,) ambient Pauli coefficients
        gamma_D_C = float(np.linalg.norm(n_coeffs))

        M = E @ N                                                         # (n_ops, null_dim) orthonormal basis
                                                                            # of N_{C,M} in ambient coords: M^T M = I
        q_norm_sq = np.einsum("ij,ij->i", M, M)                           # (n_ops,) = ||P_{N_{C,M}} e_j||^2 = ||q_j||^2
        exact_score = np.where(q_norm_sq > 1e-16, r_DC_coef ** 2 / np.maximum(q_norm_sq, 1e-300), 0.0)
    else:
        r_DC_coef = np.zeros(n_ops)
        gamma_D_C = 0.0
        exact_score = np.zeros(n_ops)

    R_manifold = gamma_D_C / gamma_D if gamma_D > 0 else float("nan")
    return ManifoldObservabilityResult(gamma_D=gamma_D, gamma_D_C=gamma_D_C, R_manifold=R_manifold,
                                        tangent_rank=rank, null_dim=null_dim, g_coef=g_coef,
                                        r_DC_coef=r_DC_coef, exact_score=exact_score)


def self_test():
    print("Running self-test (manifold/restricted_observability)...")
    import sys
    from pathlib import Path
    # candidates/oracle live in the sibling this repository/ (this project's shared,
    # already-audited Pauli-pool/oracle infrastructure -- not duplicated here).
    from candidates.pauli_pool import build_pool

    rng = np.random.RandomState(0)
    a = 2
    d = 2 ** a
    pool = build_pool(a)

    # random valid density matrix
    z = rng.randn(d, d) + 1j * rng.randn(d, d)
    rho = z @ z.conj().T
    rho = rho / np.trace(rho).real

    # 1. Full-rank tangent (jac spans everything, i.e. T_rho C = ambient):
    #    gamma_{D,C} should equal gamma_D exactly (no restriction at all).
    n_full = d * d  # deliberately overcomplete/random basis of Hermitian d x d matrices
    jac_full = np.zeros((d, d, n_full), dtype=complex)
    for i in range(n_full):
        h = rng.randn(d, d) + 1j * rng.randn(d, d)
        h = h + h.conj().T
        jac_full[:, :, i] = h
    res_full = compute_R_manifold(rho, jac_full, pool)

    g = -hermitian_log(rho)
    g_coef = pauli_coefficients(g, pool.all_matrices)
    gamma_D_direct = float(np.linalg.norm(g_coef[pool.offdiag_idx]))
    assert abs(res_full.gamma_D - gamma_D_direct) < 1e-8
    assert abs(res_full.gamma_D_C - res_full.gamma_D) < 1e-6, (
        f"full-rank tangent should recover ambient gamma_D exactly, got "
        f"gamma_D_C={res_full.gamma_D_C} vs gamma_D={res_full.gamma_D}")
    print(f"  [check 1] full-rank (unrestricted) tangent recovers ambient gamma_D exactly "
          f"(gamma_D={res_full.gamma_D:.4f}, gamma_D_C={res_full.gamma_D_C:.4f})")

    # 2. Cross-check gamma_D against oracle.entropy_target.score_candidates's
    #    independently-implemented gamma0 (the module already trusted and
    #    used throughout this project).
    from oracle.entropy_target import score_candidates
    oracle_gamma0 = score_candidates(rho, pool.offdiag_matrices).gamma0
    assert abs(res_full.gamma_D - oracle_gamma0) < 1e-8, (
        f"gamma_D mismatch vs oracle.entropy_target: {res_full.gamma_D} vs {oracle_gamma0}")
    print(f"  [check 2] gamma_D matches oracle.entropy_target.score_candidates's gamma0 exactly "
          f"({res_full.gamma_D:.6f} vs {oracle_gamma0:.6f})")

    # 3. Zero tangent (jac all-zero): gamma_{D,C} must be 0 (nothing reachable).
    jac_zero = np.zeros((d, d, 5), dtype=complex)
    res_zero = compute_R_manifold(rho, jac_zero, pool)
    assert res_zero.tangent_rank == 0 and res_zero.gamma_D_C == 0.0
    print(f"  [check 3] zero tangent space gives gamma_D_C=0, tangent_rank=0")

    # 4. Tangent spanning EXACTLY the measured (diagonal) subspace: every
    #    reachable direction is fully visible to the measurement, so the
    #    null space within T_rho C must be empty and gamma_D_C=0.
    n_diag = len(pool.diag_idx)
    jac_diag_only = np.zeros((d, d, n_diag), dtype=complex)
    for k, idx in enumerate(pool.diag_idx):
        jac_diag_only[:, :, k] = pool.all_matrices[idx]
    res_diag = compute_R_manifold(rho, jac_diag_only, pool)
    assert res_diag.null_dim == 0, f"expected null_dim=0 when T_rho C = measured span, got {res_diag.null_dim}"
    assert res_diag.gamma_D_C == 0.0
    print(f"  [check 4] tangent space = measured span exactly -> null_dim=0, gamma_D_C=0 "
          f"(rank={res_diag.tangent_rank})")

    # 5. Numerical check of guide Theorem A/B's underlying mechanics -- not
    #    just the linear-algebra tautology, but the actual physics: is
    #    r_DC (a) truly measurement-invisible, (b) a real, nonzero
    #    first-order direction the diagnostic IS sensitive to (this is
    #    exactly the "blind spot" Theorem B closes), and (c) does adding it
    #    as ONE new observable actually drive the completed gamma_{D,C}
    #    to ~0, per Theorem B's own claim?
    a2 = 3
    d2 = 2 ** a2
    pool2 = build_pool(a2)
    z2 = rng.randn(d2, d2) + 1j * rng.randn(d2, d2)
    rho2 = z2 @ z2.conj().T
    rho2 = rho2 / np.trace(rho2).real
    n_params2 = 20  # << ambient dim 63, guarantees a nontrivial null space
    jac2 = np.zeros((d2, d2, n_params2), dtype=complex)
    for i in range(n_params2):
        h = rng.randn(d2, d2) + 1j * rng.randn(d2, d2)
        h = h + h.conj().T
        h = h - np.trace(h).real / d2 * np.eye(d2)   # traceless, like a real d(rho)/d(theta_i)
        jac2[:, :, i] = h
    res2 = compute_R_manifold(rho2, jac2, pool2)
    assert res2.gamma_D_C > 1e-3, "test setup should yield a nontrivial blind direction"

    # (a) r_DC measurement-invisible: Tr[S0_k @ r_DC_matrix] ~= 0 for every diag op.
    r_DC_matrix = np.tensordot(res2.r_DC_coef, pool2.all_matrices, axes=(0, 0)) / np.sqrt(d2)
    diag_overlaps = np.array([np.trace(pool2.all_matrices[k] @ r_DC_matrix).real for k in pool2.diag_idx])
    assert np.abs(diag_overlaps).max() < 1e-8, f"r_DC should be measurement-invisible, got {np.abs(diag_overlaps).max()}"
    print(f"  [check 5a] r_DC is exactly measurement-invisible (max|Tr[S0_k @ r_DC]|="
          f"{np.abs(diag_overlaps).max():.2e})")

    # (b) r_DC is a real, nonzero diagnostic-sensitive direction:
    #     dD_rho(r_DC) = <g, r_DC> should equal ||r_DC||^2 = gamma_D_C^2 exactly
    #     (both are just <g_coef, r_DC_coef> computed two ways).
    dD_via_g = float(res2.g_coef @ res2.r_DC_coef)
    assert abs(dD_via_g - res2.gamma_D_C ** 2) < 1e-8, (
        f"dD_rho(r_DC) should equal gamma_D_C^2: {dD_via_g} vs {res2.gamma_D_C**2}")
    print(f"  [check 5b] r_DC is diagnostic-sensitive: <g,r_DC>={dD_via_g:.4f} == "
          f"gamma_D_C^2={res2.gamma_D_C**2:.4f} -- exactly the blind-but-real direction "
          f"Theorem B's completion is supposed to close")

    # (c) Theorem B: adding ONE new observable B* propto r_DC to the measured
    #     set should drive the COMPLETED gamma_{D,C} to ~0. Recompute the
    #     tangent basis E and the augmented null space directly (duplicating
    #     compute_R_manifold's internal SVD steps -- this is a validation
    #     probe, not meant to grow the public API).
    jac2_moved = np.moveaxis(jac2, -1, 0)
    jac2_coef = pauli_coefficients(jac2_moved, pool2.all_matrices).T
    U2, S2, _ = np.linalg.svd(jac2_coef, full_matrices=False)
    rank2 = int((S2 > 1e-8 * S2.max()).sum())
    E2 = U2[:, :rank2]
    b_star_coef = res2.r_DC_coef / np.linalg.norm(res2.r_DC_coef)
    A_before = E2[pool2.diag_idx, :]
    A_after = np.vstack([A_before, (b_star_coef @ E2)[None, :]])   # append B*'s row to the measurement map
    _, Sa2, Vha2 = np.linalg.svd(A_after, full_matrices=True)
    rank_a2 = int((Sa2 > 1e-6 * max(Sa2.max(), 1e-300)).sum())
    N2_after = Vha2.T[:, rank_a2:]
    c_proj2 = E2.T @ res2.g_coef
    gamma_D_C_after = float(np.linalg.norm(N2_after.T @ c_proj2)) if N2_after.shape[1] > 0 else 0.0
    assert gamma_D_C_after < 1e-6, (
        f"Theorem B: adding B*=r_DC/||r_DC|| should drive completed gamma_D_C to ~0, got {gamma_D_C_after}")
    print(f"  [check 5c] Theorem B confirmed numerically: gamma_D_C before completion="
          f"{res2.gamma_D_C:.4f}, AFTER adding B*=r_DC as one new observable="
          f"{gamma_D_C_after:.2e} (~0, first-order-complete)")

    # 6. `extra_measured_idx` parameter (used by run_experiment27's
    #    multi-sample robust completion check): augmenting with a REAL
    #    single pool candidate (not the abstract r_DC direction) should
    #    only ever shrink gamma_D_C (adding a measurement can't make the
    #    model less observable), and augmenting with the FULL offdiag set
    #    should drive it to exactly 0 (nothing left unmeasured).
    j_pick = pool2.offdiag_idx[int(np.argmax(res2.r_DC_coef[pool2.offdiag_idx] ** 2))]
    res_aug_one = compute_R_manifold(rho2, jac2, pool2, extra_measured_idx=[j_pick])
    assert res_aug_one.gamma_D_C <= res2.gamma_D_C + 1e-9, (
        f"adding a measurement should not increase gamma_D_C: {res_aug_one.gamma_D_C} vs {res2.gamma_D_C}")
    res_aug_all = compute_R_manifold(rho2, jac2, pool2, extra_measured_idx=list(pool2.offdiag_idx))
    assert res_aug_all.gamma_D_C < 1e-8, f"measuring everything should leave gamma_D_C~0, got {res_aug_all.gamma_D_C}"
    print(f"  [check 6] extra_measured_idx: adding the single best real candidate shrinks "
          f"gamma_D_C {res2.gamma_D_C:.4f} -> {res_aug_one.gamma_D_C:.4f}; adding ALL offdiag "
          f"candidates drives it to {res_aug_all.gamma_D_C:.2e} (~0)")

    # 7. `extra_measured_vectors` (Ablation B, README Follow-up 50): the
    #    public/reusable path must reproduce check 5c's hand-rolled
    #    computation exactly (same B*=r_DC/||r_DC|| direction).
    res_aug_vec = compute_R_manifold(rho2, jac2, pool2, extra_measured_vectors=[b_star_coef])
    assert abs(res_aug_vec.gamma_D_C - gamma_D_C_after) < 1e-10, (
        f"extra_measured_vectors should reproduce check 5c's hand-rolled result exactly: "
        f"{res_aug_vec.gamma_D_C} vs {gamma_D_C_after}")
    # combining WITH extra_measured_idx (both params at once) must also work
    res_aug_both = compute_R_manifold(rho2, jac2, pool2, extra_measured_idx=[j_pick],
                                       extra_measured_vectors=[b_star_coef])
    assert res_aug_both.gamma_D_C <= res_aug_one.gamma_D_C + 1e-9
    print(f"  [check 7] extra_measured_vectors reproduces check 5c exactly "
          f"({res_aug_vec.gamma_D_C:.2e} vs {gamma_D_C_after:.2e}); combining with "
          f"extra_measured_idx also works ({res_aug_both.gamma_D_C:.2e})")

    # 8. SI~A motivates real Pauli-coefficient coordinates by noting that an
    #    ordinary COMPLEX SVD of the vectorized Hermitian-matrix-valued
    #    Jacobian does not, in general, return individually-Hermitian
    #    singular vectors. This check asks the sharper, operationally
    #    relevant question directly: does the ORTHOGONAL PROJECTOR built
    #    from that complex SVD actually give a different (wrong) answer
    #    when applied to a genuinely Hermitian target, or does it coincide
    #    with the real-coefficient method's projector? On a random
    #    synthetic tangent space and a random Hermitian target, compare the
    #    two projections directly in the ambient matrix space.
    # `pool.all_matrices` spans the TRACELESS Hermitian operators only
    # (4^a-1 of them, the identity excluded); jac columns are drawn
    # traceless here too, matching the physical case (Tr[rho_A]=1 for
    # every theta, so d(rho_A)/d(theta_i) is automatically traceless) --
    # a nonzero-trace component would compare two different ambient
    # spaces, not test the complex-SVD question itself.
    n8 = rng.randint(3, d * d - 1)
    jac8 = np.zeros((d, d, n8), dtype=complex)
    for i in range(n8):
        h = rng.randn(d, d) + 1j * rng.randn(d, d)
        h = h + h.conj().T
        h = h - (np.trace(h).real / d) * np.eye(d)
        jac8[:, :, i] = h
    jac8_coef = pauli_coefficients(np.moveaxis(jac8, -1, 0), pool.all_matrices).T
    U8r, S8r, _ = np.linalg.svd(jac8_coef, full_matrices=False)
    rank8 = int((S8r > 1e-8 * S8r.max()).sum())
    E8 = U8r[:, :rank8]
    target = -hermitian_log(rho)
    target = target - (np.trace(target).real / d) * np.eye(d)
    target_coef = pauli_coefficients(target, pool.all_matrices)
    real_proj_coef = E8 @ (E8.T @ target_coef)

    jac8_vec = jac8.reshape(d * d, n8)
    U8c, S8c, _ = np.linalg.svd(jac8_vec, full_matrices=False)
    rank8c = int((S8c > 1e-8 * S8c.max()).sum())
    Uc8 = U8c[:, :rank8c]
    target_vec = target.reshape(d * d)
    complex_proj_mat = (Uc8 @ (Uc8.conj().T @ target_vec)).reshape(d, d)
    herm_defect = float(np.max(np.abs(complex_proj_mat - complex_proj_mat.conj().T)))
    complex_proj_coef = pauli_coefficients(complex_proj_mat, pool.all_matrices)
    disagreement = float(np.linalg.norm(complex_proj_coef - real_proj_coef))
    assert herm_defect < 1e-8, (
        f"complex-SVD projection of a Hermitian target should stay Hermitian "
        f"(defect={herm_defect:.2e})")
    assert disagreement < 1e-6 * max(np.linalg.norm(real_proj_coef), 1e-300), (
        f"complex-SVD projector should coincide with the real-coefficient "
        f"projector on a Hermitian target (disagreement={disagreement:.2e})")
    print(f"  [check 8] complex-SVD-derived projector, applied to a Hermitian target, "
          f"stays Hermitian (defect={herm_defect:.1e}) and numerically coincides with the "
          f"real-Pauli-coefficient projector (disagreement={disagreement:.1e}) -- individual "
          f"complex singular vectors need not be Hermitian, but the projector built from them "
          f"is basis-independent and gives the physically correct answer regardless")

    # 9. Section 4's exact_score field, cross-checked against run_experiment52's
    #    independently-implemented ||q_j|| computation on a real trained state
    #    already known to be a numerator-only-vs-exact disagreement case
    #    (bloodmnist_seed44_before, README Follow-up 65/96): the exact rule
    #    should pick candidate 119 (XZYI), not the numerator-only rule's 125
    #    (XZZY), and the implied ||q_j|| range should match Experiment 52's
    #    persisted 0.558577-0.760157 to high precision.
    pool4 = build_pool(4)
    art44 = np.load(Path(__file__).resolve().parents[1] / "results" / "manifold_artifacts"
                     / "bloodmnist_seed44_before.npz")
    res44 = compute_R_manifold(art44["rho_A"], art44["jac"], pool4)
    offdiag44 = np.array(pool4.offdiag_idx)
    j_exact = int(offdiag44[np.argmax(res44.exact_score[offdiag44])])
    j_numer = int(offdiag44[np.argmax(res44.r_DC_coef[offdiag44] ** 2)])
    q_norm44 = np.sqrt(res44.r_DC_coef[offdiag44] ** 2 /
                        np.maximum(res44.exact_score[offdiag44], 1e-300))
    assert j_exact == 119 and j_numer == 125, (
        f"exact_score/numerator-only disagreement case regressed: "
        f"j_exact={j_exact} (expected 119), j_numer={j_numer} (expected 125)")
    assert abs(q_norm44.min() - 0.558577) < 1e-5 and abs(q_norm44.max() - 0.760157) < 1e-5, (
        f"||q_j|| range regressed vs Experiment 52: got [{q_norm44.min():.6f}, "
        f"{q_norm44.max():.6f}], expected [0.558577, 0.760157]")
    print(f"  [check 9] exact_score field matches Experiment 52's independent ||q_j|| "
          f"computation exactly (range [{q_norm44.min():.6f}, {q_norm44.max():.6f}]), and "
          f"picks the same disagreeing candidate (exact={j_exact} vs numerator-only={j_numer}) "
          f"on the known bloodmnist_seed44_before disagreement case")

    print("Self-test PASSED.\n")


if __name__ == "__main__":
    self_test()
