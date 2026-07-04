"""cli.py — interactive terminal transport.

The local terminal user is identified as user_id "local"; the default
config.yaml lists that ID as a cli owner, so the person at the keyboard
has owner privileges (including tools) out of the box. Remote transports
get no such default — Telegram/Discord owners must be listed explicitly.
"""

from rich.console import Console
from rich.panel import Panel

from milly_agent.transports.base import Transport

LOCAL_USER_ID = "local"


class CLITransport(Transport):
    name = "cli"

    def run(self) -> None:
        console = Console()
        session_id = str(self.transport_config.get("session_id", "cli"))
        principal = self.principal(LOCAL_USER_ID, display_name="terminal user")

        console.print(
            Panel(
                "milly-agent — type a message, or /exit to quit.",
                title="Milly",
                border_style="cyan",
            )
        )
        while True:
            try:
                text = console.input("[bold cyan]you>[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not text:
                continue
            if text.lower() in {"/exit", "/quit"}:
                break
            reply = self.agent.handle_message(principal, text, session_id)
            console.print(f"[bold magenta]milly>[/bold magenta] {reply}")

        console.print("[dim]bye.[/dim]")
