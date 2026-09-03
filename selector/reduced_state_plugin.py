"""Route A: reduced-state plug-in selector (guide Section 11, Route A).

Algorithm: linear-inversion reconstruct rho_A from pilot Pauli estimates,
project to PSD/trace-1, regularize toward the maximally mixed state,
take -log of the regularized estimate, and rank candidates by the
resulting score. This is the low-risk baseline the guide flags as
"global-subtomographic, but not the final goal" (Route B, direct moment
scoring without ever forming rho_hat, is the more novel follow-up).

HARD CONSTRAINT (guide Section 18): this module must never receive the
true density matrix, the true entropy gradient, the true unmeasured
Pauli coefficients, or the true eigenvalues. Its only inputs are pilot
measurement estimates and the (state-independent) Pauli operator
definitions.
"""
from dataclasses import dataclass

import numpy as np


def _project_psd_trace1(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + rho.conj().T)
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        d = rho.shape[0]
        return np.eye(d, dtype=complex) / d
    w = w / total
    return (v * w) @ v.conj().T


def _hermitian_log(rho: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 1e-14, None)
    return (v * np.log(w)) @ v.conj().T


@dataclass
class SelectorOutput:
    rho_hat: np.ndarray
    a_hat: np.ndarray
    delta_hat: np.ndarray
    order: np.ndarray
    j_hat: int


def select(chat: np.ndarray, all_matrices: np.ndarray, offdiag_local_idx,
           d: int, reg_lambda: float = 0.01) -> SelectorOutput:
    """
    chat: pilot estimates for every nontrivial Pauli operator (length 4^a-1).
    all_matrices: matching Pauli matrices, shape (4^a-1, d, d).
    offdiag_local_idx: indices (into `all_matrices`/`chat`) of the candidate
        (off-diagonal, C0) directions to be ranked.
    """
    rho_hat = (np.eye(d, dtype=complex) +
               np.einsum("m,mij->ij", chat.astype(complex), all_matrices)) / d
    rho_hat = _project_psd_trace1(rho_hat)
    rho_reg = (1 - reg_lambda) * rho_hat + reg_lambda * np.eye(d, dtype=complex) / d

    g_hat = -_hermitian_log(rho_reg)
    offdiag_matrices = all_matrices[offdiag_local_idx]
    a_hat = np.einsum("ij,mji->m", g_hat, offdiag_matrices) / np.sqrt(d)
    delta_hat = np.abs(a_hat) ** 2
    order = np.argsort(-delta_hat)
    j_hat = int(order[0])
    return SelectorOutput(rho_hat=rho_reg, a_hat=a_hat, delta_hat=delta_hat,
                           order=order, j_hat=j_hat)
