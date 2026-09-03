"""Grouped (qubit-wise-commuting) Pauli pilot measurement simulator
(README Follow-up 68; guide/model_restricted_observability_theory_
revision_plan.md Section 14.1: "至少增加一个: qubit-wise commuting
grouping ... 比较对象: 1. naive per-operator allocation; 2. grouped
tomography; 3. classical shadows").

`measurement_simulator/pauli_pilot.py`'s existing "direct tomography"
baseline allocates one DEDICATED measurement setting per operator
(4^a-1 settings for n_a=4 -- 255). This wastes measurement budget:
several offdiag Pauli operators are qubit-wise commuting (share the
same per-qubit measurement basis) and can be read out from the SAME
circuit execution. This module implements the standard fix real VQE
experiments already use: partition the operator pool into
qubit-wise-commuting (QWC) groups, allocate the pilot budget across
GROUPS (not individual operators), and directly sample-mean-estimate
every operator in a group from the same shots -- no random-basis
correction factor needed (unlike classical shadows), since every shot
within a group setting genuinely used that exact basis throughout.

Canonical grouping convention (matches candidates.pauli_pool / this
project's own Ablation B `_qwc_group_containing`): operator with
support S and per-qubit type tau on S is assigned to the setting that
matches tau on S and pads every identity (non-support) position with Z
-- the same convention used throughout this project for "what a real
device gets for free alongside a specific target Pauli." This gives
each operator (diag AND offdiag alike -- matching simulate_pilot's and
simulate_shadow_pilot's own convention of estimating the full
4^{n_a}-1 pool, needed for a fair three-way budget comparison) exactly
one canonical group across all 3^{n_a} settings: every diag operator
(support type in {I,Z} only) Z-pads to the SAME all-Z setting (which
therefore alone carries all 2^{n_a}-1 diag operators), while every
offdiag operator pads to one of the remaining 3^{n_a}-1 settings that
has at least one non-Z symbol.

Usage: `simulate_grouped_pilot(rho_true, pool, n_pilot, rng)` returns a
`chat`-style array (same convention as `measurement_simulator.pauli_
pilot.simulate_pilot` and `measurement_simulator_shadow.simulate_shadow_
pilot`), a drop-in replacement for either.
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
class GroupedPilotResult:
    chat: np.ndarray            # (4^a-1,) real, unbiased estimate of Tr[P_m @ rho_true]
    shots_requested: int
    n_groups: int                # 3^n_a - 1 settings actually used
    shots_per_group: int


def _canonical_settings(pool):
    """Returns: settings (all 3^{n_a} basis tuples), and for each setting
    index, the list of (global_pool_index, support_positions) operators
    canonically assigned to it. Every nontrivial operator (diag AND
    offdiag alike, matching simulate_pilot/simulate_shadow_pilot's own
    convention of estimating the full 4^{n_a}-1 pool, not offdiag only,
    for a fair three-way comparison) is assigned to EXACTLY one setting:
    its own type on its support, Z-padded on identity positions. Every
    diag operator (support type in {I,Z} only) canonically pads to the
    all-Z setting; every offdiag operator pads to a setting with at
    least one non-Z symbol."""
    n_a = pool.a
    settings = list(product((1, 2, 3), repeat=n_a))
    setting_to_idx = {b: i for i, b in enumerate(settings)}
    members = [[] for _ in settings]
    for gi, t in enumerate(pool.all_tuples):
        support = [q for q, x in enumerate(t) if x != 0]
        canonical = tuple(t[q] if q in support else 3 for q in range(n_a))  # pad identity with Z
        si = setting_to_idx[canonical]
        members[si].append((gi, support))
    return settings, members


def simulate_grouped_pilot(rho_true: np.ndarray, pool, n_pilot: int,
                            rng: np.random.Generator) -> GroupedPilotResult:
    n_a, d = pool.a, pool.d
    settings, members = _canonical_settings(pool)
    n_groups = len(settings)
    n_ops = len(pool.all_tuples)

    if n_pilot <= 0 or n_groups == 0:
        return GroupedPilotResult(chat=np.zeros(n_ops), shots_requested=n_pilot,
                                   n_groups=n_groups, shots_per_group=0)

    shots_per_group = n_pilot // n_groups
    chat = np.zeros(n_ops)
    if shots_per_group <= 0:
        return GroupedPilotResult(chat=chat, shots_requested=n_pilot,
                                   n_groups=n_groups, shots_per_group=0)

    for si, basis in enumerate(settings):
        group_members = members[si]
        if not group_members:
            continue
        U = _basis_unitary(basis)
        rho_rot = U @ rho_true @ U.conj().T
        probs = np.clip(rho_rot.diagonal().real, 0.0, None)
        probs = probs / probs.sum()
        outcomes = rng.choice(d, size=shots_per_group, p=probs)
        shifts = np.arange(n_a - 1, -1, -1)
        bits = (outcomes[:, None] >> shifts[None, :]) & 1
        signs = 1 - 2 * bits  # (shots_per_group, n_a)
        for gi, support in group_members:
            prod_signs = np.prod(signs[:, support], axis=1) if support else np.ones(shots_per_group)
            chat[gi] = prod_signs.mean()

    return GroupedPilotResult(chat=chat, shots_requested=n_pilot,
                               n_groups=n_groups, shots_per_group=shots_per_group)


def self_test():
    print("Running self-test (measurement_simulator_grouped)...")
    import sys
    from pathlib import Path
    from candidates.pauli_pool import build_pool

    rng = np.random.RandomState(0)
    a = 3
    d = 2 ** a
    pool = build_pool(a)
    z = rng.randn(d, d) + 1j * rng.randn(d, d)
    rho = z @ z.conj().T
    rho = rho / np.trace(rho).real
    true_expect = np.einsum("ij,mji->m", rho, pool.all_matrices).real

    settings, members = _canonical_settings(pool)
    assert len(settings) == 3 ** a, f"expected {3**a} settings, got {len(settings)}"
    covered = set()
    for grp in members:
        for gi, _ in grp:
            covered.add(gi)
    assert covered == set(range(len(pool.all_tuples))), "every operator (diag+offdiag) must be covered exactly once"
    total_assignments = sum(len(g) for g in members)
    assert total_assignments == len(pool.all_tuples), (
        f"canonical assignment should be a partition (each operator exactly once): "
        f"{total_assignments} assignments vs {len(pool.all_tuples)} operators")
    all_z = tuple([3] * a)
    all_z_members = members[settings.index(all_z)]
    assert {gi for gi, _ in all_z_members} == set(pool.diag_idx), (
        "the all-Z setting must be assigned exactly the diag operators")
    print(f"  [check 1] {len(settings)}=3^{a} settings partition all {len(pool.all_tuples)} "
          f"operators exactly once (no overlap, no gap); all-Z setting covers exactly "
          f"the {len(pool.diag_idx)} diag operators")

    rng_np = np.random.default_rng(1)
    n_draws, n_pilot = 400, 2000
    chats = np.stack([simulate_grouped_pilot(rho, pool, n_pilot, rng_np).chat for _ in range(n_draws)])
    mean_chat = chats.mean(axis=0)
    max_err = np.abs(mean_chat - true_expect).max()
    mc_se = chats.std(axis=0).max() / np.sqrt(n_draws)
    assert max_err < 8 * mc_se + 1e-3, (
        f"grouped chat should be unbiased on all ops: max|mean-true|={max_err:.4f}, "
        f"8x MC-SE={8*mc_se:.4f}")
    print(f"  [check 2] grouped chat is unbiased on all {len(pool.all_tuples)} operators "
          f"(diag+offdiag) over {n_draws} draws "
          f"(max|mean-true|={max_err:.4f}, 8x MC-SE={8*mc_se:.4f})")

    big = simulate_grouped_pilot(rho, pool, 2_000_000, np.random.default_rng(2))
    rho_hat = (np.eye(d, dtype=complex) + np.einsum("m,mij->ij", big.chat.astype(complex),
                                                       pool.all_matrices)) / d
    err_rho = np.abs(rho_hat - rho).max()
    assert err_rho < 0.05, f"reconstructed rho_hat should match rho closely at large n_pilot: max|err|={err_rho:.4f}"
    print(f"  [check 3] uses {big.n_groups}=3^{a} grouped settings "
          f"(vs. direct tomography's {4**a-1}=4^{a}-1 dedicated settings, "
          f"vs. classical shadows' {3**a}=3^{a} random settings); "
          f"full rho reconstruction from chat matches rho (max|err|={err_rho:.4f})")

    print("Self-test PASSED.\n")


if __name__ == "__main__":
    self_test()
