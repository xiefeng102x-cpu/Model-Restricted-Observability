"""Pauli operator pool for an `a`-qubit target subsystem, per guide Section 19.

S0 = complete diagonal Pauli span (I/Z only on every factor) -- the
"already measured" computational-basis record.
C0 = off-diagonal Pauli strings (contains at least one X or Y factor) --
the candidate directions a selector can choose to open next.

Counts are exactly 2^a - 1 (S0, nontrivial) and 4^a - 2^a (C0), matching
the guide's worked table (a=2..6 -> |C0| = 12, 56, 240, 992, 4032).
"""
from dataclasses import dataclass
from functools import reduce
from itertools import product

import numpy as np

_PAULI = {
    0: np.eye(2, dtype=complex),
    1: np.array([[0, 1], [1, 0]], dtype=complex),
    2: np.array([[0, -1j], [1j, 0]], dtype=complex),
    3: np.array([[1, 0], [0, -1]], dtype=complex),
}
_LABEL = {0: "I", 1: "X", 2: "Y", 3: "Z"}


def _string_label(idx_tuple):
    return "".join(_LABEL[i] for i in idx_tuple)


def _pauli_matrix(idx_tuple):
    return reduce(np.kron, (_PAULI[i] for i in idx_tuple))


@dataclass
class PauliPool:
    a: int
    d: int
    diag_idx: list          # indices into `all_tuples` classified as S0 (diagonal)
    offdiag_idx: list       # indices into `all_tuples` classified as C0 (candidates)
    all_tuples: list        # every nontrivial (non-identity) index tuple, length 4^a-1
    all_matrices: np.ndarray  # shape (4^a-1, d, d), complex128
    all_labels: list

    @property
    def diag_matrices(self):
        return self.all_matrices[self.diag_idx]

    @property
    def offdiag_matrices(self):
        return self.all_matrices[self.offdiag_idx]

    @property
    def offdiag_labels(self):
        return [self.all_labels[i] for i in self.offdiag_idx]

    @property
    def offdiag_global_to_local(self) -> dict:
        """Maps a GLOBAL candidate index (position in `all_tuples`/
        `all_matrices`, the full 4^a-1 array -- what `oracle.score_candidates`
        was NOT built from, and what `selector.reduced_state_plugin.select`'s
        `offdiag_local_idx` parameter actually expects despite its name) to
        its LOCAL position within `offdiag_idx`/`offdiag_matrices` (what
        `oracle.delta_j`/`oracle.j_star` and Route B's candidate_tuples
        arrays are indexed by). Any code building a restricted candidate
        subset for Route A from oracle-derived local indices must map
        through this before calling `select_route_a` -- mixing the two
        conventions silently retrieves the wrong Pauli matrices."""
        return {g: i for i, g in enumerate(self.offdiag_idx)}

    def contiguous_offdiag_idx(self) -> list:
        """Off-diagonal candidates whose non-identity support forms an
        unbroken run of consecutive qubit positions along the natural
        chain 0..a-1 -- e.g. support {1,2} qualifies, {0,2} does not.

        This is a purely structural filter (a function of `a` alone, no
        state or oracle access) motivated by real circuits whose own
        entangling gates and any post-hoc intervention both follow a
        ring/nearest-neighbour topology (this repository's trained classifiers
        and its ZZ-null-unitary intervention both use RING_PAIRS =
        [(q,(q+1)%n_qubits) ...]): theory (the known entangling
        topology), not data from the unknown state, predicts contiguous
        directions are more likely informative. Empirically checked
        (this repository diagnostics) against both random states (a small
        generic bias toward contiguous winners exists even with no
        physical topology at all) and this repository's real trained states
        (a stronger elevation, 90% vs. a 68.3% base rate at a=4, n=10 --
        suggestive, not statistically conclusive at that sample size).
        """
        out = []
        for i in self.offdiag_idx:
            support = [q for q, x in enumerate(self.all_tuples[i]) if x != 0]
            if not support:
                continue
            if support == list(range(min(support), max(support) + 1)):
                out.append(i)
        return out


def build_pool(a: int) -> PauliPool:
    d = 2**a
    all_tuples = [t for t in product(range(4), repeat=a) if any(x != 0 for x in t)]
    diag_idx, offdiag_idx = [], []
    for i, t in enumerate(all_tuples):
        is_diagonal = all(x in (0, 3) for x in t)
        (diag_idx if is_diagonal else offdiag_idx).append(i)

    n_diag_expected = 2**a - 1
    n_offdiag_expected = 4**a - 2**a
    assert len(diag_idx) == n_diag_expected, (len(diag_idx), n_diag_expected)
    assert len(offdiag_idx) == n_offdiag_expected, (len(offdiag_idx), n_offdiag_expected)

    matrices = np.stack([_pauli_matrix(t) for t in all_tuples], axis=0)
    labels = [_string_label(t) for t in all_tuples]
    return PauliPool(
        a=a, d=d, diag_idx=diag_idx, offdiag_idx=offdiag_idx,
        all_tuples=all_tuples, all_matrices=matrices, all_labels=labels,
    )
