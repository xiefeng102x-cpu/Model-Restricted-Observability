"""Deterministic seed derivation, per guide Section 32.

Independent seed streams (state / pilot measurement / selector / evaluation)
are derived from SHA256(experiment_id, unit_id, round_id, role) so that any
single stream can be regenerated without re-running the others.
"""
import hashlib


def derive_seed(experiment_id: str, unit_id: str, round_id: str, role: str) -> int:
    payload = "|".join([experiment_id, unit_id, round_id, role]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32 - 1)
