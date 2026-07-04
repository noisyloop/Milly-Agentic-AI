"""test_smoke.py — end-to-end smoke test for milly-agent (was smoke_test.py).

Uses a scripted fake LLM (no Ollama server required) to verify:
  1. Tool execution        — a scripted tool call writes a file in the sandbox
  2. Owner gating          — a guest's tool call is denied, no file is written
  3. Stranger denial       — unknown users never reach the model
  4. Signed-memory persistence — history survives reload; tampering is caught
  5. Iteration cap         — a tool-happy model is stopped at the cap

Runnable both ways:
  python -m pytest tests/
  python tests/test_smoke.py
"""

import json
import sys
import tempfile
from pathlib import Path

from milly_agent.agent import Agent
from milly_agent.authz import AuthzPolicy, Principal
from milly_agent.core.audit import AuditLog
from milly_agent.core.guardian import Guardian
from milly_agent.core.memory import Memory, MemoryIntegrityError
from milly_agent.tools import ToolRegistry

OWNER = Principal(transport="cli", user_id="owner-1", display_name="the owner")
GUEST = Principal(transport="cli", user_id="guest-1", display_name="a guest")
STRANGER = Principal(transport="cli", user_id="rando-9", display_name="a stranger")

AUTHZ_CONFIG = {
    "owners": {"cli": ["owner-1"]},
    "guests": {"cli": ["guest-1"]},
    "allow_strangers": False,
    "owner_only_tools": True,
}


class FakeLLM:
    """Scripted LLM: returns canned replies in order, counts calls."""

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls = 0

    def chat(self, messages: list[dict]) -> str:
        self.calls += 1
        if not self.script:
            return "(fake llm: script exhausted)"
        return self.script.pop(0)


def make_agent(data_dir: Path, script: list[str], max_iterations: int = 3):
    """Build an Agent wired exactly like cli.build_agent, minus Ollama/RAG."""
    guardian = Guardian(
        {
            "enabled": True,
            "sensitivity": "medium",
            "max_input_length": 4000,
            "custom_patterns_file": str(data_dir / "no_custom_patterns.txt"),
        }
    )
    memory = Memory(memory_dir=str(data_dir / "memory"))
    audit = AuditLog(log_dir=str(data_dir / "logs"))
    tools = ToolRegistry(data_dir / "workspace")
    authz = AuthzPolicy(AUTHZ_CONFIG)
    llm = FakeLLM(script)
    agent = Agent(
        config={"max_tool_iterations": max_iterations, "default_model": "fake"},
        llm=llm,
        guardian=guardian,
        memory=memory,
        audit=audit,
        tools=tools,
        authz=authz,
    )
    return agent, llm


def _tool_call(name: str, **args) -> str:
    return json.dumps({"tool": name, "args": args})


# ----------------------------------------------------------------------
# 1. Tool execution
# ----------------------------------------------------------------------

def test_tool_execution():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        agent, llm = make_agent(
            data_dir,
            [
                _tool_call("write_file", path="notes/hello.txt", content="hi from milly"),
                "Done — I wrote notes/hello.txt for you.",
            ],
        )
        reply = agent.handle_message(OWNER, "Please save a hello note.", "s-tools")

        written = data_dir / "workspace" / "notes" / "hello.txt"
        assert written.is_file(), "tool did not create the file in the workspace"
        assert written.read_text(encoding="utf-8") == "hi from milly"
        assert "Done" in reply
        assert llm.calls == 2  # one tool round-trip + one final answer


# ----------------------------------------------------------------------
# 2. Owner gating — guests may chat but tools are owner-only
# ----------------------------------------------------------------------

def test_owner_gating():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        agent, llm = make_agent(
            data_dir,
            [
                _tool_call("write_file", path="sneaky.txt", content="guest was here"),
                "Sorry, I wasn't able to write that file.",
            ],
        )
        reply = agent.handle_message(GUEST, "Write me a file please.", "s-guest")

        assert llm.calls == 2, "guest chat should reach the model"
        assert not (data_dir / "workspace" / "sneaky.txt").exists(), (
            "tool ran for a non-owner — owner gating failed"
        )
        assert "Sorry" in reply

        events = agent.audit.get_session_events("s-guest")
        assert any(e["event"] == "tool_denied" for e in events)
        assert not any(e["event"] == "tool_executed" for e in events)


# ----------------------------------------------------------------------
# 3. Stranger denial — unknown users never reach the model
# ----------------------------------------------------------------------

def test_stranger_denial():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        agent, llm = make_agent(data_dir, ["should never be returned"])
        reply = agent.handle_message(STRANGER, "hello?", "s-stranger")

        assert llm.calls == 0, "stranger input reached the model"
        assert "Access denied" in reply
        assert not (data_dir / "memory" / "s-stranger.json").exists(), (
            "denied conversation must not be persisted"
        )
        events = agent.audit.get_session_events("s-stranger")
        assert any(e["event"] == "authz_message_denied" for e in events)


# ----------------------------------------------------------------------
# 4. Signed-memory persistence — reload verifies, tampering is rejected
# ----------------------------------------------------------------------

def test_signed_memory_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        agent, _ = make_agent(data_dir, ["Nice to meet you!"])
        agent.handle_message(OWNER, "Hi, I'm the owner.", "s-mem")

        # Fresh Memory instance (same dir/key) must verify and load history.
        memory2 = Memory(memory_dir=str(data_dir / "memory"))
        history = memory2.load("s-mem")
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["content"] == "Hi, I'm the owner."
        assert history[1]["content"] == "Nice to meet you!"

        # The on-disk envelope is signed…
        session_file = data_dir / "memory" / "s-mem.json"
        stored = json.loads(session_file.read_text(encoding="utf-8"))
        assert set(stored) == {"data", "sig"}

        # …and tampering with it must be detected on load.
        stored["data"] = stored["data"].replace("owner", "hacker")
        session_file.write_text(json.dumps(stored), encoding="utf-8")
        try:
            memory2.load("s-mem")
        except MemoryIntegrityError:
            pass
        else:
            raise AssertionError("tampered session file passed HMAC verification")


# ----------------------------------------------------------------------
# 5. Iteration cap — tool-happy model is cut off
# ----------------------------------------------------------------------

def test_iteration_cap():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        cap = 3
        agent, llm = make_agent(
            data_dir,
            [_tool_call("list_files")] * 10,  # never produces a final answer
            max_iterations=cap,
        )
        reply = agent.handle_message(OWNER, "List files forever.", "s-cap")

        assert llm.calls == cap, f"expected exactly {cap} LLM calls, got {llm.calls}"
        assert "iteration cap" in reply
        events = agent.audit.get_session_events("s-cap")
        assert any(e["event"] == "iteration_cap_reached" for e in events)


# ----------------------------------------------------------------------
# Direct execution: python tests/test_smoke.py
# ----------------------------------------------------------------------

def main() -> int:
    tests = [
        test_tool_execution,
        test_owner_gating,
        test_stranger_denial,
        test_signed_memory_persistence,
        test_iteration_cap,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as e:  # noqa: BLE001 — report and continue
            failures += 1
            print(f"FAIL  {test.__name__}: {e}")
        else:
            print(f"PASS  {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} smoke tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
