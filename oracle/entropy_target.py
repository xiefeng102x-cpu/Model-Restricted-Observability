"""Ground-truth (oracle-only) scoring for the entropy target D(rho)=S(rho_A).

Per guide Section 1 and Section 9: interior gradient G_S(rho) = -log(rho),
candidate score Delta_j(rho) = |<b_j, g_D(rho)>_HS|^2 for normalized
b_j = P_j / sqrt(d).

This module is oracle-only: everything here consumes the true density
matrix directly and must NOT be imported by selector/ (guide Section 18).
"""
from dataclasses import dataclass

import numpy as np


def hermitian_log(rho: np.ndarray, eig_floor: float = 1e-14) -> np.ndarray:
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, eig_floor, None)
    return (v * np.log(w)) @ v.conj().T


@dataclass
class OracleScores:
    a_j: np.ndarray        # complex, shape (n_candidates,)
    delta_j: np.ndarray    # real, shape (n_candidates,)
    order: np.ndarray      # candidate indices sorted by delta_j descending
    j_star: int            # candidate index of the top score (into offdiag pool)
    margin: float          # Delta_(1) - Delta_(2)
    gamma0: float           # sqrt(sum of all candidate Delta_j) = blind-gradient norm


def score_candidates(rho: np.ndarray, offdiag_matrices: np.ndarray) -> OracleScores:
    d = rho.shape[0]
    g = -hermitian_log(rho)
    a_j = np.einsum("ij,mji->m", g, offdiag_matrices) / np.sqrt(d)
    delta_j = np.abs(a_j) ** 2
    order = np.argsort(-delta_j)
    j_star = int(order[0])
    margin = float(delta_j[order[0]] - delta_j[order[1]]) if len(order) > 1 else float("nan")
    gamma0 = float(np.sqrt(delta_j.sum()))
    return OracleScores(a_j=a_j, delta_j=delta_j, order=order, j_star=j_star,
                         margin=margin, gamma0=gamma0)
