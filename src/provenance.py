"""Cross-platform hashing helpers for experiment provenance.

Git may materialize tracked text files with LF or CRLF line endings depending
on the checkout configuration. Provenance hashes must identify logical text
content, not that platform-specific representation.
"""

from __future__ import annotations

import hashlib
import os


_TEXT_SUFFIXES = frozenset({
    ".csv", ".json", ".md", ".py", ".rst", ".txt", ".xml", ".yaml", ".yml",
})


def sha256_file(path):
    """Return a SHA-256 digest stable across LF and CRLF text checkouts."""
    suffix = os.path.splitext(os.fspath(path))[1].lower()
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        if suffix not in _TEXT_SUFFIXES:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()

        pending_cr = False
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if pending_cr:
                chunk = b"\r" + chunk
                pending_cr = False
            if chunk.endswith(b"\r"):
                chunk = chunk[:-1]
                pending_cr = True
            digest.update(chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        if pending_cr:
            digest.update(b"\n")
    return digest.hexdigest()
