# milly-agent

Milly is a **local-first agentic AI assistant** built on the Milly security
core. She runs against a local [Ollama](https://ollama.com) model, can use a
small set of sandboxed file tools, and talks to you over your terminal,
Telegram, or Discord — with authorization, prompt-injection screening, signed
memory, and audit logging enforced on every message.

## Quickstart

```bash
# 1. Install (editable, from the repo root)
pip install -e .

# 2. Make sure Ollama is running and the model is pulled
ollama pull llama3.2

# 3. Talk to Milly in your terminal
milly-agent --transport cli
```

Other transports:

```bash
# Telegram (set your bot token first)
export TELEGRAM_BOT_TOKEN=123456:ABC...
milly-agent --transport telegram

# Discord (optional dependency — install the extra first)
pip install -e ".[discord]"
export DISCORD_BOT_TOKEN=...
milly-agent --transport discord
```

Useful flags: `--config /path/to/config.yaml` (default: `$MILLY_AGENT_CONFIG`
or `./config.yaml`), `--model NAME` to override `default_model`.

## Package layout

```
milly_agent/
├── agent.py          # agent loop: authz → Guardian → LLM → tools → memory
├── authz.py          # owner / guest / stranger policy per transport
├── tools.py          # sandboxed workspace file tools
├── cli.py            # console entry point (`milly-agent`)
├── core/             # vendored Milly security core (unchanged logic)
│   ├── guardian.py   #   input/output security, injection detection
│   ├── memory.py     #   HMAC-signed persistent session history
│   ├── audit.py      #   structured security event log
│   └── rag.py        #   safe document ingestion + TF-IDF retrieval
└── transports/       # cli, telegram, discord channels
```

## Runtime data

All runtime state lives under a single base directory, `data_dir` in
`config.yaml` (default `./data`, resolved relative to the config file — not
your shell's working directory):

```
data/
├── docs/        # drop documents here for RAG retrieval
├── logs/        # security.log — structured audit events (hashes, never content)
├── memory/      # HMAC-signed session history + RAG index
└── workspace/   # the ONLY directory the agent's file tools can touch
```

`data/logs`, `data/memory`, and `data/workspace` are gitignored (runtime
state); the `.gitkeep` markers keep the directories present in the repo.

## Authorization

Access is keyed on `(transport, user id)` in `config.yaml`:

- **owners** — full access, including tool execution. The terminal user
  (`cli: ["local"]`) is an owner by default.
- **guests** — may chat, but tools stay owner-only while
  `owner_only_tools: true`.
- **strangers** — everyone else; denied outright unless
  `allow_strangers: true` (and never given tools).

For Telegram/Discord, list the numeric user IDs under
`authz.owners.telegram` / `authz.owners.discord`.

## Security model

Every message passes through the vendored Milly security core:

- **Guardian** — length limits, OWASP-LLM prompt-injection patterns
  (sensitivity `low`/`medium`/`high`, plus your own regexes in
  `custom_patterns.txt`), character sanitization, and output filtering.
- **Memory** — session history is HMAC-SHA256 signed; tampered files are
  rejected on load.
- **AuditLog** — denials, flags, tool runs, and cap events are logged to
  `data/logs/security.log` as JSON (input hashes only, never content).
- **Tool sandbox** — file tools resolve paths strictly inside
  `data/workspace`; absolute paths and traversal are rejected.
- **Iteration cap** — at most `max_tool_iterations` LLM calls per user
  message, so a confused model can't loop forever.

## Development

```bash
pip install -e .
python -m pytest tests/            # or: python tests/test_smoke.py
```

The smoke test uses a scripted fake LLM — no Ollama server needed — and
covers tool execution, owner gating, stranger denial, signed-memory
persistence, and the iteration cap.

## License

MIT — see [LICENSE](LICENSE).
