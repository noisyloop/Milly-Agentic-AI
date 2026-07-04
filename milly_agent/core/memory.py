"""
memory.py — HMAC-signed persistent conversation history.

Storage format (per session file):
    {"data": "<json-encoded history>", "sig": "<hmac-sha256 hex>"}

Security properties:
  - Files created with mode 0o600 (owner read/write only).
  - HMAC key generated with secrets.token_bytes on first run, stored in memory/.key.
  - Signature verified on every load; tampered or corrupted files are rejected.
  - User messages written to history only after successful model response.
  - History capped at max_history * 2 entries (user + assistant pairs).
"""

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Optional


class MemoryIntegrityError(Exception):
    """Raised when a session file fails HMAC verification."""


class Memory:
    def __init__(self, memory_dir: str = "memory", max_history: int = 50):
        self.dir = Path(memory_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_history  # max user+assistant pairs
        self._key: bytes = self._load_or_create_key()

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def _key_path(self) -> Path:
        return self.dir / ".key"

    def _load_or_create_key(self) -> bytes:
        path = self._key_path()
        if path.exists():
            # Enforce restricted permissions on existing key file
            try:
                path.chmod(0o600)
            except OSError:
                pass  # best-effort on restricted filesystems
            with open(path, "rb") as f:
                return f.read()
        key = secrets.token_bytes(32)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key)
        return key

    # ------------------------------------------------------------------
    # HMAC helpers
    # ------------------------------------------------------------------

    def _sign(self, data: str) -> str:
        return hmac.new(self._key, data.encode("utf-8"), hashlib.sha256).hexdigest()

    def _verify(self, data: str, sig: str) -> bool:
        expected = self._sign(data)
        return hmac.compare_digest(expected, sig)

    # ------------------------------------------------------------------
    # Session paths
    # ------------------------------------------------------------------

    def _session_path(self, session_id: str) -> Path:
        # Sanitize session ID to safe filename characters
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
        if not safe:
            raise ValueError(f"Invalid session ID: '{session_id}' produces empty filename")
        return self.dir / f"{safe}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, session_id: str) -> list[dict]:
        """
        Load and verify a session.
        Raises FileNotFoundError if session does not exist.
        Raises MemoryIntegrityError on tamper/corruption.
        """
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found.")

        with open(path, "r", encoding="utf-8") as f:
            stored = json.load(f)

        if not isinstance(stored, dict):
            raise MemoryIntegrityError(
                f"Session '{session_id}' is not a valid session file "
                "(unexpected format). Use '/session new' to start fresh."
            )

        data: str = stored.get("data", "")
        sig: str = stored.get("sig", "")

        if not self._verify(data, sig):
            raise MemoryIntegrityError(
                f"Session '{session_id}' failed integrity check. "
                "The history file may have been modified or corrupted. "
                "Use '/session new' to start fresh."
            )

        return json.loads(data)

    def save(self, session_id: str, history: list[dict]) -> None:
        """Sign and save session history. Creates file with mode 0o600."""
        data = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
        sig = self._sign(data)
        stored = {"data": data, "sig": sig}

        path = self._session_path(session_id)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(stored, f)

    def append(
        self,
        session_id: str,
        history: list[dict],
        role: str,
        content: str,
    ) -> list[dict]:
        """
        Return a new history list with the message appended.
        Trims to max_turns * 2 entries if needed.
        Does NOT save — caller saves only after model responds successfully.
        """
        updated = list(history)
        updated.append({"role": role, "content": content})
        cap = self.max_turns * 2
        if len(updated) > cap:
            updated = updated[-cap:]
        return updated

    def clear(self, session_id: str) -> list[dict]:
        """Clear history and persist the empty state."""
        self.save(session_id, [])
        return []

    def list_sessions(self) -> list[str]:
        """Return saved session IDs.

        The memory/ directory also holds non-session JSON (e.g. the RAG
        index), so a bare glob would surface bogus "sessions". Only files
        carrying the signed-envelope shape (``data`` + ``sig`` keys) are
        treated as sessions.
        """
        sessions: list[str] = []
        for p in self.dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    stored = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(stored, dict) and "data" in stored and "sig" in stored:
                sessions.append(p.stem)
        return sorted(sessions)

    def delete_session(self, session_id: str) -> None:
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
