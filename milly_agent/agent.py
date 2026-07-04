"""agent.py — the milly-agent conversation/tool loop.

Wires together AuthzPolicy → Guardian → LLM → tools → Memory → AuditLog.

The LLM is injected as any object exposing ``chat(messages) -> str`` where
``messages`` is a list of ``{"role": ..., "content": ...}`` dicts. Production
uses OllamaLLM below; tests use a scripted fake.

Tool-call protocol (kept model-agnostic so it works with plain local models):
the model requests a tool by replying with ONLY a JSON object

    {"tool": "<name>", "args": {...}}

optionally wrapped in a ```json fenced block. Anything else is treated as the
final answer for the turn. The loop runs at most ``max_tool_iterations`` LLM
calls per user message; if the model is still asking for tools at the cap,
the turn ends with a capped notice instead of another call.
"""

import json
from typing import Optional

from milly_agent.authz import AuthzPolicy, Principal
from milly_agent.core.audit import AuditLog
from milly_agent.core.guardian import Guardian
from milly_agent.core.memory import Memory, MemoryIntegrityError
from milly_agent.core.rag import RAG
from milly_agent.tools import ToolError, ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are Milly, a helpful local AI agent.\n"
    "Be helpful, honest, and concise."
)

_TOOL_PROTOCOL = """
You can use tools. To call a tool, reply with ONLY a JSON object of the form:
{"tool": "<tool name>", "args": {<arguments>}}
Do not add any other text to a tool-call reply. After each tool call you will
receive the tool's result and may call another tool or answer the user.
Available tools:
"""


class OllamaLLM:
    """Thin Ollama chat client. Import is lazy so tests never need ollama."""

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 temperature: float = 0.7):
        self.model = model
        self.host = host
        self.temperature = temperature

    def chat(self, messages: list[dict]) -> str:
        import ollama

        client = ollama.Client(host=self.host)
        response = client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]


def parse_tool_call(text: str) -> Optional[tuple[str, dict]]:
    """Return (tool_name, args) if the reply is a tool call, else None."""
    t = text.strip()
    if t.startswith("```"):
        # Strip a fenced block: ```json\n{...}\n```
        lines = t.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            t = "\n".join(lines[1:-1]).strip()
    if not (t.startswith("{") and t.endswith("}")):
        return None
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("tool"), str):
        return None
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    return obj["tool"], args


class Agent:
    def __init__(
        self,
        config: dict,
        llm,
        guardian: Guardian,
        memory: Memory,
        audit: AuditLog,
        tools: ToolRegistry,
        authz: AuthzPolicy,
        rag: Optional[RAG] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.config = config or {}
        self.llm = llm
        self.guardian = guardian
        self.memory = memory
        self.audit = audit
        self.tools = tools
        self.authz = authz
        self.rag = rag
        self.system_prompt = system_prompt
        self.max_iterations: int = max(1, int(self.config.get("max_tool_iterations", 5)))
        self.model_name: str = str(self.config.get("default_model", ""))

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _full_system_prompt(self) -> str:
        parts = [self.system_prompt, _TOOL_PROTOCOL.rstrip()]
        for spec in self.tools.specs():
            args = ", ".join(f"{k}: {v}" for k, v in spec["args"].items()) or "none"
            parts.append(f"- {spec['name']}: {spec['description']} (args: {args})")
        return "\n".join(parts)

    def _load_history(self, session_id: str) -> list[dict]:
        try:
            return self.memory.load(session_id)
        except FileNotFoundError:
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_message(self, principal: Principal, text: str, session_id: str) -> str:
        """Process one inbound user message and return the reply text."""
        # 1. Authorization gate — strangers never reach the model.
        decision = self.authz.check_message(principal)
        if not decision.allowed:
            self.audit.log(
                session_id,
                "authz_message_denied",
                model=self.model_name,
                transport=principal.transport,
                user_id=str(principal.user_id),
                reason=decision.reason,
            )
            return "Access denied: you are not authorized to talk to this agent."

        # 2. Guardian input checks.
        guard = self.guardian.check(text)
        if guard.blocked:
            self.audit.log(
                session_id,
                "input_blocked",
                model=self.model_name,
                input_hash=guard.input_hash,
                reason=guard.reason,
            )
            return f"Input blocked by Guardian: {guard.reason}"
        if guard.flagged:
            self.audit.log(
                session_id,
                "input_flagged",
                model=self.model_name,
                input_hash=guard.input_hash,
                pattern=guard.pattern,
            )
        user_text = guard.sanitized_input

        # 3. Signed history + optional RAG context.
        try:
            history = self._load_history(session_id)
        except MemoryIntegrityError as e:
            self.audit.log(session_id, "memory_integrity_failure", reason=str(e))
            return f"Session refused: {e}"

        messages: list[dict] = [{"role": "system", "content": self._full_system_prompt()}]
        if self.rag is not None and self.rag.doc_count > 0:
            context = self.rag.format_context(self.rag.query(user_text))
            if context:
                messages.append({"role": "system", "content": context})
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        # 4. Agent loop — bounded by max_tool_iterations LLM calls.
        final_reply: Optional[str] = None
        for _ in range(self.max_iterations):
            reply = self.llm.chat(messages)
            call = parse_tool_call(reply)
            if call is None:
                final_reply = reply
                break

            tool_name, tool_args = call
            messages.append({"role": "assistant", "content": reply})
            tool_decision = self.authz.check_tool(principal, tool_name)
            if not tool_decision.allowed:
                self.audit.log(
                    session_id,
                    "tool_denied",
                    model=self.model_name,
                    transport=principal.transport,
                    user_id=str(principal.user_id),
                    tool=tool_name,
                    reason=tool_decision.reason,
                )
                result = f"[tool denied: {tool_decision.reason}]"
            else:
                try:
                    result = self.tools.execute(tool_name, tool_args)
                    self.audit.log(
                        session_id,
                        "tool_executed",
                        model=self.model_name,
                        transport=principal.transport,
                        user_id=str(principal.user_id),
                        tool=tool_name,
                    )
                except ToolError as e:
                    self.audit.log(
                        session_id,
                        "tool_error",
                        model=self.model_name,
                        tool=tool_name,
                        reason=str(e),
                    )
                    result = f"[tool error: {e}]"
            messages.append({"role": "tool", "content": result})

        if final_reply is None:
            self.audit.log(
                session_id,
                "iteration_cap_reached",
                model=self.model_name,
                cap=self.max_iterations,
            )
            final_reply = (
                f"[stopped: reached the tool iteration cap "
                f"({self.max_iterations}) without a final answer]"
            )

        final_reply = self.guardian.filter_output(final_reply)

        # 5. Persist signed history only after a successful turn.
        history = self.memory.append(session_id, history, "user", user_text)
        history = self.memory.append(session_id, history, "assistant", final_reply)
        self.memory.save(session_id, history)

        return final_reply
