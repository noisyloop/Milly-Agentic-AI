"""milly_agent.transports — message transports for milly-agent.

Each transport turns an external channel (terminal, Telegram, Discord)
into (Principal, text) pairs for the Agent and delivers replies back.
"""

from milly_agent.transports.base import Transport


def make_transport(name: str, agent, config: dict) -> Transport:
    """Instantiate a transport by name. Imports are lazy so optional
    dependencies (discord.py) are only required when actually used."""
    if name == "cli":
        from milly_agent.transports.cli import CLITransport

        return CLITransport(agent, config)
    if name == "telegram":
        from milly_agent.transports.telegram import TelegramTransport

        return TelegramTransport(agent, config)
    if name == "discord":
        from milly_agent.transports.discord import DiscordTransport

        return DiscordTransport(agent, config)
    raise ValueError(f"unknown transport: {name!r} (expected cli, telegram, or discord)")


__all__ = ["Transport", "make_transport"]
