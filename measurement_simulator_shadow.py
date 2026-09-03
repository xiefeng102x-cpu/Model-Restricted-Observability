"""Classical-shadow-based pilot measurement simulator (README Follow-up
55; manuscript Discussion item 3, "P0-1": a genuinely measurement-
sublinear pilot, previously flagged as not implemented).

this repository's existing `measurement_simulator/pauli_pilot.py` implements
DIRECT per-operator tomography: the shot budget is split EVENLY across
all `4^a-1` nontrivial Pauli settings (one dedicated measurement circuit
per operator), so touching every operator at all requires allocating
shots to all of them. This module implements the alternative the
manuscript's own Discussion flagged as untested: a Huang-Kueng-Preskill
(2020, arXiv:2002.08953) random-Pauli classical-shadow protocol, where
EVERY shot uses ONE randomly chosen product-Pauli measurement basis
(`3^a` possible settings, not `4^a-1` dedicated ones), and the SAME pool
of shots produces an unbiased estimate for EVERY one of the `4^a-1`
target operators simultaneously -- no operator is ever "not touched" by
a given shot budget the way direct tomography's per-operator split can
leave under-sampled operators at low budgets.

Output convention matches `simulate_pilot`'s own `chat` field exactly
(unbiased estimate of Tr[P_m @ rho_true] per nontrivial Pauli operator,
same ordering as `pool.all_matrices`) -- a drop-in replacement, so all
downstream machinery (`selector.reduced_state_plugin.select`,
`manifold.restricted_observability.compute_R_manifold`) is reused
unchanged.

Derivation (single-qubit factor algebra): a single random-Pauli shadow
snapshot from measuring qubit i in basis b_i and obtaining outcome
s_i in {0,1} contributes the classical-shadow factor 3|psi_i><psi_i|-I,
where |psi_i> is the b_i-eigenstate of eigenvalue (1-2*s_i). For a
target Pauli P_m (a tuple over n_a qubits),
Tr[P_m @ (tensor of per-qubit snapshot factors)] equals 1 at every
qubit where P_m is identity, equals 3*(1-2*s_i) at every qubit where
P_m's own factor EXACTLY matches the measured basis b_i, and equals 0
EXACTLY (single-qubit Pauli orthogonality: an eigenstate of one Pauli
has zero expectation under either of the other two) at any qubit where
P_m's factor is a DIFFERENT nonidentity Pauli than the measured basis.
So a single shot contributes a nonzero estimate to P_m only when its
random basis draw matches P_m's own support pattern -- the same
qubit-wise-commuting (QWC) compatibility structure Ablation B
(Experiment 40) used for "the group compatible with one specific
winning Pauli," generalized here to "the group compatible with an
arbitrary random basis draw."
"""
from dataclasses import dataclass
from functools import reduce
from itertools import product

import numpy as np

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.diag([1, -1j]).astype(complex)
_ROT = {1: _H, 2: _H @ _SDG, 3: np.eye(2, dtype=complex)}   # 1=X,2=Y,3=Z (pool.py convention)


def _basis_unitary(basis_tuple) -> np.ndarray:
    return reduce(np.kron, (_ROT[b] for b in basis_tuple))


@dataclass
class ShadowPilotResult:
    chat: np.ndarray            # (4^a-1,) real, unbiased estimate of Tr[P_m @ rho_true], matches pool.all_tuples order
    shots_requested: int
    n_basis_settings: int       # 3**n_a random-basis settings, vs. direct tomography's 4**n_a-1 dedicated ones


def simulate_shadow_pilot(rho_true: np.ndarray, pool, n_pilot: int,
                           rng: np.random.Generator) -> ShadowPilotResult:
    n_a, d = pool.a, pool.d
    basis_tuples = list(product((1, 2, 3), repeat=n_a))
    n_settings = len(basis_tuples)
    n_ops = len(pool.all_tuples)

    if n_pilot <= 0:
        return ShadowPilotResult(chat=np.zeros(n_ops), shots_requested=n_pilot, n_basis_settings=n_settings)

    probs_by_setting = []
    for basis in basis_tuples:
        U = _basis_unitary(basis)
        rho_rot = U @ rho_true @ U.conj().T
        p = np.clip(rho_rot.diagonal().real, 0.0, None)
        p = p / p.sum()
        probs_by_setting.append(p)

    compat_by_setting = []
    for basis in basis_tuples:
        members = []
        for gi, t in enumerate(pool.all_tuples):
            if all(x == 0 or x == basis[q] for q, x in enumerate(t)):
                support = [q for q, x in enumerate(t) if x != 0]
                members.append((gi, support))
        compat_by_setting.append(members)

    chat_sum = np.zeros(n_ops)
    setting_idx = rng.integers(0, n_settings, size=n_pilot)
    counts = np.bincount(setting_idx, minlength=n_settings)

    for si, count in enumerate(counts):
        if count == 0:
            continue
        outcomes = rng.choice(d, size=int(count), p=probs_by_setting[si])
        shifts = np.arange(n_a - 1, -1, -1)
        bits = (outcomes[:, None] >> shifts[None, :]) & 1     # (count, n_a)
        signs = 1 - 2 * bits                                   # (count, n_a) in {-1,+1}
        for gi, support in compat_by_setting[si]:
            weight = len(support)
            prod_signs = np.prod(signs[:, support], axis=1) if support else np.ones(count)
            chat_sum[gi] += (3.0 ** weight) * prod_signs.sum()

    chat = chat_sum / n_pilot
    return ShadowPilotResult(chat=chat, shots_requested=n_pilot, n_basis_settings=n_settings)


def self_test():
    print("Running self-test (measurement_simulator_shadow)...")
    import sys
    from pathlib import Path
    from candidates.pauli_pool import build_pool

    # 0. rotation-convention check: U_b must map the b-eigenstate of
    #    eigenvalue +1 to the computational |0> state (up to global phase).
    eigvecs = {
        1: np.array([1, 1]) / np.sqrt(2),          # X, eigenvalue +1
        2: np.array([1, 1j]) / np.sqrt(2),         # Y, eigenvalue +1
        3: np.array([1, 0]),                       # Z, eigenvalue +1
    }
    for b, vec in eigvecs.items():
        rotated = _ROT[b] @ vec
        overlap = abs(rotated[0])
        assert abs(overlap - 1.0) < 1e-10, f"basis {b}: U_b @ (+1 eigenstate) should be |0> up to phase, got {rotated}"
    print("  [check 0] single-qubit rotation unitaries map each Pauli's +1-eigenstate to |0> (up to phase)")

    rng = np.random.RandomState(0)
    a = 3
    d = 2 ** a
    pool = build_pool(a)
    z = rng.randn(d, d) + 1j * rng.randn(d, d)
    rho = z @ z.conj().T
    rho = rho / np.trace(rho).real

    true_expect = np.einsum("ij,mji->m", rho, pool.all_matrices).real

    # 1. unbiasedness: average chat over many independent shadow draws at a
    #    modest per-draw budget should converge to the true expectation values.
    rng_np = np.random.default_rng(1)
    n_draws, n_pilot = 400, 2000
    chats = np.stack([simulate_shadow_pilot(rho, pool, n_pilot, rng_np).chat for _ in range(n_draws)])
    mean_chat = chats.mean(axis=0)
    max_err = np.abs(mean_chat - true_expect).max()
    mc_se = chats.std(axis=0).max() / np.sqrt(n_draws)
    assert max_err < 8 * mc_se + 1e-3, (
        f"shadow chat should be unbiased: max|mean_chat - true|={max_err:.4f}, "
        f"8x Monte Carlo SE={8*mc_se:.4f}")
    print(f"  [check 1] shadow chat is unbiased over {n_draws} draws "
          f"(max|mean-true|={max_err:.4f}, 8x MC-SE={8*mc_se:.4f})")

    # 2. large-budget single draw: reconstructed rho_hat should converge to
    #    the true rho_A, using the SAME inversion formula selector.
    #    reduced_state_plugin.select uses internally.
    big = simulate_shadow_pilot(rho, pool, 2_000_000, np.random.default_rng(2))
    rho_hat = (np.eye(d, dtype=complex) + np.einsum("m,mij->ij", big.chat.astype(complex),
                                                       pool.all_matrices)) / d
    err_rho = np.abs(rho_hat - rho).max()
    assert err_rho < 0.05, f"large-budget shadow reconstruction should approach true rho_A, max err={err_rho:.4f}"
    print(f"  [check 2] large-budget (n=2e6) shadow reconstruction approaches true rho_A "
          f"(max element error={err_rho:.4f})")

    # 3. n_basis_settings must be 3^a, distinct from direct tomography's 4^a-1.
    assert big.n_basis_settings == 3 ** a
    print(f"  [check 3] uses {big.n_basis_settings}=3^{a} random basis settings "
          f"(direct tomography would need {4**a-1}=4^{a}-1 dedicated settings)")

    print("Self-test PASSED.\n")


if __name__ == "__main__":
    self_test()
