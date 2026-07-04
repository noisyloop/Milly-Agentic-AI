"""telegram.py — Telegram bot transport (long polling, stdlib + requests).

Token comes from transports.telegram.token in config.yaml or the
TELEGRAM_BOT_TOKEN environment variable. Authorization is enforced by
the agent's AuthzPolicy against the sender's numeric Telegram user ID
(authz.owners.telegram / authz.guests.telegram in config.yaml).
"""

import os
import time

import requests

from milly_agent.transports.base import Transport

_POLL_TIMEOUT_S = 30
_RETRY_DELAY_S = 5


class TelegramTransport(Transport):
    name = "telegram"

    def _token(self) -> str:
        token = self.transport_config.get("token") or os.environ.get(
            "TELEGRAM_BOT_TOKEN", ""
        )
        if not token:
            raise RuntimeError(
                "Telegram token missing: set transports.telegram.token in "
                "config.yaml or the TELEGRAM_BOT_TOKEN environment variable."
            )
        return token

    def _api(self, token: str, method: str, **params) -> dict:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/{method}",
            json=params,
            timeout=_POLL_TIMEOUT_S + 10,
        )
        resp.raise_for_status()
        return resp.json()

    def run(self) -> None:
        token = self._token()
        offset = 0
        print("milly-agent: Telegram transport polling for updates…")
        while True:
            try:
                data = self._api(
                    token,
                    "getUpdates",
                    offset=offset,
                    timeout=_POLL_TIMEOUT_S,
                    allowed_updates=["message"],
                )
            except requests.RequestException as e:
                print(f"telegram poll error: {e}; retrying in {_RETRY_DELAY_S}s")
                time.sleep(_RETRY_DELAY_S)
                continue

            for update in data.get("result", []):
                offset = max(offset, update["update_id"] + 1)
                message = update.get("message") or {}
                text = message.get("text")
                sender = message.get("from") or {}
                chat = message.get("chat") or {}
                if not text or "id" not in sender or "id" not in chat:
                    continue

                principal = self.principal(
                    sender["id"], display_name=sender.get("username", "")
                )
                session_id = f"telegram-{chat['id']}"
                reply = self.agent.handle_message(principal, text, session_id)
                try:
                    self._api(token, "sendMessage", chat_id=chat["id"], text=reply)
                except requests.RequestException as e:
                    print(f"telegram send error: {e}")
