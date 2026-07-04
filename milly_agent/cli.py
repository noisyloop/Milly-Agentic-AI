"""cli.py — console entry point for milly-agent (was run.py).

Installed as the ``milly-agent`` command via [project.scripts].

Path model: every runtime directory (docs/, logs/, memory/, workspace/)
lives under a single base data directory. The base comes from ``data_dir``
in config.yaml (default ``./data``); a relative value is resolved against
the directory containing the config file — NOT the process CWD — so the
command works no matter where it is invoked from. With no config file at
all, the CWD is the anchor of last resort.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

from milly_agent.agent import Agent, DEFAULT_SYSTEM_PROMPT, OllamaLLM
from milly_agent.authz import AuthzPolicy
from milly_agent.core.audit import AuditLog
from milly_agent.core.guardian import Guardian
from milly_agent.core.memory import Memory
from milly_agent.core.rag import RAG
from milly_agent.tools import ToolRegistry
from milly_agent.transports import make_transport

CONFIG_ENV_VAR = "MILLY_AGENT_CONFIG"


def find_config(explicit: str | None = None) -> Path | None:
    """Locate config.yaml: --config flag, then $MILLY_AGENT_CONFIG, then CWD."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd() / "config.yaml")

    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.is_file():
            return candidate.resolve()
    if explicit:
        raise FileNotFoundError(f"config file not found: {explicit}")
    return None


def load_config(config_path: Path | None) -> tuple[dict, Path]:
    """Return (config dict, base dir that relative paths resolve against)."""
    if config_path is None:
        return {}, Path.cwd()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config file is not a mapping: {config_path}")
    return config, config_path.parent


def resolve_data_dir(config: dict, base_dir: Path) -> Path:
    """Resolve the runtime data directory (data_dir key, default ./data)."""
    raw = Path(str(config.get("data_dir", "data"))).expanduser()
    data_dir = raw if raw.is_absolute() else base_dir / raw
    return data_dir.resolve()


def _load_system_prompt(config: dict, base_dir: Path) -> str:
    prompt_file = base_dir / "system_prompt.txt"
    if prompt_file.is_file():
        text = prompt_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    return str(config.get("system_prompt", DEFAULT_SYSTEM_PROMPT))


def build_agent(config: dict, base_dir: Path, model_override: str | None = None) -> Agent:
    """Assemble the agent with all runtime dirs anchored at data_dir."""
    data_dir = resolve_data_dir(config, base_dir)
    docs_dir = data_dir / "docs"
    logs_dir = data_dir / "logs"
    memory_dir = data_dir / "memory"
    workspace_dir = data_dir / "workspace"
    for d in (docs_dir, logs_dir, memory_dir, workspace_dir):
        d.mkdir(parents=True, exist_ok=True)

    custom_patterns = Path(
        str(config.get("custom_patterns_file", "custom_patterns.txt"))
    ).expanduser()
    if not custom_patterns.is_absolute():
        custom_patterns = base_dir / custom_patterns

    guardian = Guardian(
        {
            "enabled": config.get("guardian_enabled", True),
            "sensitivity": config.get("guardian_sensitivity", "medium"),
            "max_input_length": config.get("max_input_length", 4000),
            "custom_patterns_file": str(custom_patterns),
        }
    )

    memory_cfg = config.get("memory") or {}
    memory = Memory(
        memory_dir=str(memory_dir),
        max_history=int(memory_cfg.get("max_history", 50)),
    )

    audit = AuditLog(log_dir=str(logs_dir))

    rag_cfg = config.get("rag") or {}
    rag = None
    if rag_cfg.get("enabled", True):
        rag = RAG(
            rag_cfg,
            guardian,
            docs_dir=str(docs_dir),
            memory_dir=str(memory_dir),
        )

    tools = ToolRegistry(workspace_dir)
    authz = AuthzPolicy(config.get("authz"))

    model = model_override or str(config.get("default_model", "llama3.2"))
    llm = OllamaLLM(
        model=model,
        host=str(config.get("ollama_host", "http://localhost:11434")),
        temperature=float(config.get("temperature", 0.7)),
    )

    agent_config = dict(config)
    agent_config["default_model"] = model
    return Agent(
        config=agent_config,
        llm=llm,
        guardian=guardian,
        memory=memory,
        audit=audit,
        tools=tools,
        authz=authz,
        rag=rag,
        system_prompt=_load_system_prompt(config, base_dir),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="milly-agent",
        description="Milly — a local-first agentic AI assistant.",
    )
    parser.add_argument(
        "--transport",
        choices=["cli", "telegram", "discord"],
        default="cli",
        help="which transport to run (default: cli)",
    )
    parser.add_argument(
        "--config",
        help=f"path to config.yaml (default: ${CONFIG_ENV_VAR} or ./config.yaml)",
    )
    parser.add_argument("--model", help="override default_model from config.yaml")
    args = parser.parse_args(argv)

    try:
        config_path = find_config(args.config)
        config, base_dir = load_config(config_path)
        agent = build_agent(config, base_dir, model_override=args.model)
        transport = make_transport(args.transport, agent, config)
        transport.run()
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"milly-agent: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
