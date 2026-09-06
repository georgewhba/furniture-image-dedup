"""Furniture Dedup — Production-grade duplicate detection for furniture images."""

from __future__ import annotations

import socket

__version__ = "1.0.0"

# Robust DNS fallback for environments with flaky local DNS (e.g. timeout on HuggingFace Hub)
try:
    import dns.resolver

    _orig_getaddrinfo = socket.getaddrinfo
    _resolver = dns.resolver.Resolver()
    _resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
    _resolver.timeout = 5.0
    _resolver.lifetime = 10.0

    def _patch_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        try:
            return _orig_getaddrinfo(host, port, family, type, proto, flags)
        except socket.gaierror:
            if isinstance(host, str) and not host.replace(".", "").isdigit():
                try:
                    answers = _resolver.resolve(host, "A")
                    ip = answers[0].to_text()
                    return _orig_getaddrinfo(ip, port, family, type, proto, flags)
                except Exception:
                    pass
            raise

    socket.getaddrinfo = _patch_getaddrinfo
except ImportError:
    pass
