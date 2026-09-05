"""Validation for the deliberately local-only ACP HTTP listener."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit


def validate_listen(value: str) -> str:
    """Return a valid loopback ``HOST:PORT`` binding or raise ``ValueError``."""
    if not value or value.strip() != value:
        raise ValueError("--listen must use HOST:PORT syntax")
    parsed = urlsplit(f"//{value}")
    if parsed.path or parsed.query or parsed.fragment or parsed.username is not None:
        raise ValueError("--listen must use HOST:PORT syntax")
    try:
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("--listen must use HOST:PORT with a valid port") from exc
    if not host or port is None or port == 0:
        raise ValueError("--listen must use HOST:PORT with a port from 1 to 65535")
    if host.casefold() != "localhost":
        try:
            address = ip_address(host)
        except ValueError as exc:
            raise ValueError("--listen accepts only a loopback IP or localhost") from exc
        if not address.is_loopback:
            raise ValueError("--listen accepts only loopback addresses")
    return value


__all__ = ["validate_listen"]
