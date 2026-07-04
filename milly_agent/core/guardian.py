"""
guardian.py — Security layer for Milly. Standalone importable.

Four checks in order:
  1. Length enforcement
  2. Prompt injection detection (OWASP LLM Top 10 patterns)
  3. Character sanitization (null bytes, control chars, RTL overrides, ANSI escapes)
  4. Output filtering (ANSI stripping, control char removal)

Usage:
    from milly_agent.core.guardian import Guardian

    g = Guardian(config)
    result = g.check(user_input)

    if result.blocked:
        print(f"Blocked: {result.reason}")
    elif result.flagged:
        print(f"Flagged ({result.pattern}): proceeding with log entry")
        sanitized = result.sanitized_input
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GuardianResult:
    blocked: bool = False
    flagged: bool = False
    reason: Optional[str] = None
    pattern: Optional[str] = None
    sanitized_input: str = ""
    input_hash: str = ""


class Guardian:
    """
    Input/output security layer.

    Covers OWASP LLM Top 10 injection categories:
      - LLM01: Prompt Injection (direct)
      - LLM02: Insecure Output Handling (output sanitization)
      - LLM06: Sensitive Information Disclosure (audit, not block)
      - Indirect injection via documents (scan_document method)
    """

    # Sensitivity tiers, ordered least → most aggressive.
    #   low    — flag only the most obvious attacks (tier "low" patterns)
    #   medium — default; everything "low" + "medium" (the original behavior)
    #   high   — everything, including "high" patterns that catch merely
    #            suspicious phrasing (hypothetical framing, encoded payloads)
    SENSITIVITY_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

    # Pattern bank: (regex, category_name, tier)
    # Each pattern targets a documented injection technique. The tier is the
    # *minimum* sensitivity at which the pattern is active: a "low" pattern
    # fires at all sensitivities, a "high" pattern fires only at "high".
    INJECTION_PATTERNS: list[tuple[str, str, str]] = [
        # Instruction override — direct instruction substitution (most obvious)
        (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|directives?)", "instruction_override", "low"),
        (r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)", "instruction_override", "low"),
        (r"forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)", "instruction_override", "low"),
        (r"override\s+(your\s+)?(instructions?|system\s+prompt|guidelines?|safety)", "instruction_override", "low"),
        (r"your\s+(new\s+)?(instructions?|prompt|system\s+prompt|rules?)\s+(are|is)\b", "instruction_override", "low"),
        (r"new\s+instructions?\s*:", "instruction_override", "low"),
        (r"from\s+now\s+on\s*(,\s*)?(you\s+)?(must|should|will|are\s+to)", "instruction_override", "medium"),
        # Persona hijacking
        (r"you\s+are\s+now\s+(?!milly\b)", "persona_override", "medium"),
        (r"pretend\s+(that\s+)?you\s+are\b", "persona_override", "medium"),
        (r"act\s+as\b", "persona_override", "medium"),
        (r"roleplay\s+as\b", "persona_override", "medium"),
        (r"simulate\s+(being\s+)?(?!milly\b)", "persona_override", "medium"),
        # System prompt injection via delimiter spoofing — authority impersonation
        (r"\bsystem\s*:\s*", "system_injection", "low"),
        (r"\[system\]", "system_injection", "low"),
        (r"<\s*system\s*>", "system_injection", "low"),
        (r"<\s*/?inst\s*>", "system_injection", "low"),
        (r"\[INST\]", "system_injection", "low"),
        (r"<<SYS>>", "system_injection", "low"),
        (r"<\|system\|>", "system_injection", "low"),
        (r"<\|im_start\|>", "system_injection", "low"),
        # Jailbreak keywords
        (r"\bjailbreak\b", "jailbreak_attempt", "medium"),
        (r"\bDAN\b", "jailbreak_attempt", "medium"),
        (r"do\s+anything\s+now", "jailbreak_attempt", "medium"),
        (r"developer\s+mode", "jailbreak_attempt", "medium"),
        (r"sudo\s+mode", "jailbreak_attempt", "medium"),
        (r"god\s+mode", "jailbreak_attempt", "medium"),
        (r"admin\s+mode", "jailbreak_attempt", "medium"),
        (r"unrestricted\s+mode", "jailbreak_attempt", "medium"),
        # Safety bypass attempts
        (r"bypass\s+(your\s+)?(safety|security|filter|restriction|guideline|ethics)", "bypass_attempt", "medium"),
        (r"(disable|turn\s+off)\s+(your\s+)?(safety|security|filter|restriction)", "bypass_attempt", "medium"),
        (r"you\s+have\s+no\s+(restrictions?|limitations?|filters?|ethics)", "bypass_attempt", "medium"),
        (r"without\s+(any\s+)?(restrictions?|limitations?|filters?|safety)", "bypass_attempt", "medium"),
        # Encoding-based evasion
        (r"base64\s*[:-]?\s*decode", "encoding_evasion", "medium"),
        (r"translate\s+the\s+following\s+(from\s+)?base64", "encoding_evasion", "medium"),
        (r"decode\s+this\s+base64", "encoding_evasion", "medium"),
        # Indirect injection markers (document-borne)
        (r"\[prompt\s+injection\]", "indirect_injection", "medium"),
        (r"<\s*injection\s*>", "indirect_injection", "medium"),
        (r"<!-- inject", "indirect_injection", "medium"),
        # Markdown-formatted injection (common in document poisoning attacks)
        (r"#{1,6}\s*(new\s+instructions?|system\s*(prompt|override))\b", "indirect_injection", "medium"),
        (r"\*{1,2}\s*(system\s*(override|prompt)|new\s+instructions?)\s*\*{1,2}", "indirect_injection", "medium"),
        # High-sensitivity only — flag merely suspicious phrasing.
        # Hypothetical / fictional framing used to coax around guardrails.
        (r"\bhypothetical(ly)?\b", "hypothetical_framing", "high"),
        (r"\bimagine\s+(that\s+)?you\b", "hypothetical_framing", "high"),
        (r"\bwhat\s+if\s+you\s+(were|could|had\s+no)\b", "hypothetical_framing", "high"),
        (r"\bfor\s+(educational|research|academic)\s+purposes\b", "hypothetical_framing", "high"),
        (r"\bin\s+a\s+(fictional|hypothetical|imaginary)\b", "hypothetical_framing", "high"),
        (r"\blet'?s\s+pretend\b", "hypothetical_framing", "high"),
        (r"\bsuppose\s+you\s+(were|are|had)\b", "hypothetical_framing", "high"),
        # Encoded / obfuscated payloads.
        (r"\brot13\b", "encoding_evasion", "high"),
        (r"\bbase64\b", "encoding_evasion", "high"),
        (r"[A-Za-z0-9+/]{24,}={0,2}", "encoding_evasion", "high"),  # long base64-like blob
        (r"(?:\\x[0-9a-fA-F]{2}){4,}", "encoding_evasion", "high"),  # \xNN hex escapes
        (r"(?:%[0-9a-fA-F]{2}){4,}", "encoding_evasion", "high"),    # URL-encoded run
    ]

    # Dangerous characters: control chars, zero-width, RTL overrides, terminal escapes
    _DANGEROUS_CHARS = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"   # ASCII control chars (keep \t \n \r)
        r"\u200b-\u200f"                         # Zero-width spaces/joiners
        r"\u202a-\u202e"                         # Bidirectional override chars
        r"\u2060-\u2064"                         # Word joiner, invisible chars
        r"\ufeff"                                # BOM
        r"\x1b"                                  # ESC — terminal escape sequences
        r"]"
    )

    # ANSI escape sequences in model output
    _ANSI_ESCAPE = re.compile(
        r"\x1b\[[0-9;]*[A-Za-z]"       # CSI sequences (colors, cursor movement)
        r"|\x1b\][^\x07]*\x07"          # OSC sequences
        r"|\x1b[()][A-Z0-9]"            # Character set designation
        r"|\x1b[^[\]()]"                # Other ESC sequences
    )

    def __init__(self, config: dict):
        self.enabled: bool = config.get("enabled", True)
        self.max_length: int = config.get("max_input_length", 4000)

        # The master `enabled` switch disables both detection layers when off.
        self.injection_detection: bool = (
            config.get("injection_detection", True) and self.enabled
        )
        self.output_sanitization: bool = (
            config.get("output_sanitization", True) and self.enabled
        )

        # Sensitivity tier controls which built-in patterns are active.
        sensitivity = str(config.get("sensitivity", "medium")).lower()
        if sensitivity not in self.SENSITIVITY_ORDER:
            sensitivity = "medium"
        self.sensitivity: str = sensitivity

        # Where users may add their own detection patterns (one regex per line).
        self.custom_patterns_file: str = config.get(
            "custom_patterns_file", "custom_patterns.txt"
        )

        # Custom user patterns always run regardless of sensitivity tier.
        custom = self._load_custom_patterns(self.custom_patterns_file)
        self.custom_pattern_count: int = len(custom)

        sens_level = self.SENSITIVITY_ORDER[self.sensitivity]
        active = [
            (pattern, name)
            for pattern, name, tier in self.INJECTION_PATTERNS
            if self.SENSITIVITY_ORDER.get(tier, 1) <= sens_level
        ]
        active.extend((pattern, name) for pattern, name in custom)

        self._compiled: list[tuple[re.Pattern, str]] = [
            (re.compile(pattern, re.IGNORECASE | re.MULTILINE), name)
            for pattern, name in active
        ]
        # Total active patterns (built-in for this tier + custom).
        self.active_pattern_count: int = len(self._compiled)

        # Lifetime detection counts for this instance (chat turns only, not doc scans)
        self._stats: dict[str, int] = {"blocked": 0, "flagged": 0, "clean": 0}

    @staticmethod
    def _load_custom_patterns(path: str) -> list[tuple[str, str]]:
        """
        Load user-supplied detection patterns from a text file.

        Format (one entry per line):
          - Blank lines and lines starting with '#' are ignored.
          - Each remaining line is a regular expression.
          - An optional human-readable name may follow the regex, separated by
            a TAB character: ``<regex>\\t<name>``. Without a name, the pattern
            is reported as "custom_pattern".

        Invalid regexes are skipped silently so a single typo cannot crash
        Milly on startup. Returns a list of (regex, name) pairs.
        """
        patterns: list[tuple[str, str]] = []
        p = Path(path)
        if not p.exists():
            return patterns
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            return patterns
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in raw:
                regex, _, name = raw.partition("\t")
                regex, name = regex.strip(), name.strip() or "custom_pattern"
            else:
                regex, name = line, "custom_pattern"
            try:
                re.compile(regex)
            except re.error:
                continue  # skip invalid regex rather than crash
            patterns.append((regex, name))
        return patterns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, user_input: str) -> GuardianResult:
        """
        Run all input checks. Returns a GuardianResult.

        Blocked  → caller must not pass input to the model.
        Flagged  → caller may proceed but should log the attempt.
        """
        result = GuardianResult()
        result.input_hash = self._hash(user_input)

        # 1. Length enforcement
        if len(user_input) > self.max_length:
            result.blocked = True
            result.reason = (
                f"Input exceeds maximum length "
                f"({len(user_input)} chars > {self.max_length} limit)"
            )
            self._stats["blocked"] += 1
            return result

        # 2. Character sanitization (before injection scan to remove evasion tricks)
        sanitized = self._strip_dangerous(user_input)

        # 3. Injection detection
        if self.injection_detection:
            for pattern, name in self._compiled:
                if pattern.search(sanitized):
                    result.flagged = True
                    result.pattern = name
                    result.reason = f"Potential prompt injection: {name}"
                    break

        result.sanitized_input = sanitized
        if result.flagged:
            self._stats["flagged"] += 1
        else:
            self._stats["clean"] += 1
        return result

    def scan_document(self, content: str) -> GuardianResult:
        """
        Scan document content for injection patterns.
        No length limit — documents are allowed to be large.
        Only scans the first 20 000 chars to bound CPU usage.
        """
        result = GuardianResult()
        result.input_hash = self._hash(content[:8192])  # hash prefix for audit

        sanitized = self._strip_dangerous(content)

        if self.injection_detection:
            sample = sanitized[:20_000]
            for pattern, name in self._compiled:
                if pattern.search(sample):
                    result.flagged = True
                    result.pattern = name
                    result.reason = f"Injection pattern in document: {name}"
                    break

        result.sanitized_input = sanitized
        return result

    def filter_output(self, text: str) -> str:
        """Sanitize model output before display."""
        if not self.output_sanitization:
            return text
        cleaned = self._ANSI_ESCAPE.sub("", text)
        cleaned = self._strip_dangerous(cleaned)
        return cleaned

    def stats(self) -> dict:
        """
        Return lifetime detection counts for this Guardian instance.

        Counts only check() calls (chat turns). scan_document() is excluded
        because document ingestion is a separate operation from chat turns.

        Returns a copy — mutating the result does not affect internal state.
        """
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def _strip_dangerous(cls, text: str) -> str:
        return cls._DANGEROUS_CHARS.sub("", text)
