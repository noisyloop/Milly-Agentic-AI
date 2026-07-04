"""discord.py — Discord bot transport (optional dependency).

Requires the ``discord`` extra:  pip install "milly-agent[discord]"

Token comes from transports.discord.token in config.yaml or the
DISCORD_BOT_TOKEN environment variable. Authorization is enforced by the
agent's AuthzPolicy against the sender's numeric Discord user ID
(authz.owners.discord / authz.guests.discord in config.yaml).
"""

import asyncio
import os

from milly_agent.transports.base import Transport


class DiscordTransport(Transport):
    name = "discord"

    def _token(self) -> str:
        token = self.transport_config.get("token") or os.environ.get(
            "DISCORD_BOT_TOKEN", ""
        )
        if not token:
            raise RuntimeError(
                "Discord token missing: set transports.discord.token in "
                "config.yaml or the DISCORD_BOT_TOKEN environment variable."
            )
        return token

    def run(self) -> None:
        try:
            import discord
        except ImportError as e:
            raise RuntimeError(
                "discord.py is not installed. Install the optional extra with: "
                "pip install 'milly-agent[discord]'"
            ) from e

        token = self._token()
        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        transport = self

        @client.event
        async def on_ready():
            print(f"milly-agent: Discord transport connected as {client.user}")

        @client.event
        async def on_message(message):
            if message.author == client.user or message.author.bot:
                return
            if not message.content:
                return
            principal = transport.principal(
                message.author.id, display_name=str(message.author)
            )
            session_id = f"discord-{message.channel.id}"
            # handle_message is synchronous (Guardian, tools, Ollama call);
            # run it off the event loop so the gateway heartbeat keeps beating.
            reply = await asyncio.to_thread(
                transport.agent.handle_message, principal, message.content, session_id
            )
            await message.channel.send(reply)

        client.run(token)
