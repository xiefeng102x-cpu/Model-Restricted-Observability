"""Finite-shot pilot measurement simulator.

Simulates a full informationally-complete Pauli pilot: the requested shot
budget N_pilot is split evenly across all 4^a-1 nontrivial Pauli settings
(both the S0/diagonal and C0/off-diagonal directions), and each setting's
expectation value is estimated from binomial +/-1 outcome sampling under
the Born rule. This is the pilot protocol Route A's reduced-state plug-in
selector (guide Section 11) consumes.

This module is the only place allowed to touch the true density matrix
for generating simulated measurement outcomes; it hands the selector
nothing but the resulting empirical estimates.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PilotResult:
    chat: np.ndarray          # empirical expectation estimate per operator, shape (4^a-1,)
    shots_requested: int
    shots_per_setting: int
    shots_actual_total: int


def simulate_pilot(rho_true: np.ndarray, all_matrices: np.ndarray, n_pilot: int,
                    rng: np.random.Generator) -> PilotResult:
    m = all_matrices.shape[0]
    shots_per_setting = n_pilot // m
    if shots_per_setting <= 0:
        return PilotResult(chat=np.zeros(m), shots_requested=n_pilot,
                            shots_per_setting=0, shots_actual_total=0)

    true_expect = np.einsum("ij,mji->m", rho_true, all_matrices).real
    p_plus = np.clip((1.0 + true_expect) / 2.0, 0.0, 1.0)
    k = rng.binomial(n=shots_per_setting, p=p_plus)
    chat = 2.0 * k / shots_per_setting - 1.0
    return PilotResult(chat=chat, shots_requested=n_pilot,
                        shots_per_setting=shots_per_setting,
                        shots_actual_total=shots_per_setting * m)
