"""authz.py — authorization policy for milly-agent.

Every inbound message carries a Principal (who is talking, over which
transport). The policy answers two questions:

  1. May this principal talk to the agent at all?   (check_message)
  2. May this principal trigger this tool?          (check_tool)

Roles:
  owner    — listed in authz.owners for the message's transport.
             Full access, including tool execution.
  guest    — listed in authz.guests for the transport.
             May chat; may not run tools while owner_only_tools is true.
  stranger — everyone else. Denied outright unless allow_strangers is true
             (and even then never allowed to run tools).
"""

from dataclasses import dataclass

ROLE_OWNER = "owner"
ROLE_GUEST = "guest"
ROLE_STRANGER = "stranger"


@dataclass(frozen=True)
class Principal:
    """Identity of the human behind a message, as seen by a transport."""

    transport: str  # "cli" | "telegram" | "discord"
    user_id: str
    display_name: str = ""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""


class AuthzPolicy:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.owners = self._id_map(cfg.get("owners"))
        self.guests = self._id_map(cfg.get("guests"))
        self.allow_strangers: bool = bool(cfg.get("allow_strangers", False))
        self.owner_only_tools: bool = bool(cfg.get("owner_only_tools", True))

    @staticmethod
    def _id_map(raw: dict | None) -> dict[str, set[str]]:
        """Normalize {transport: [ids]} with all IDs coerced to strings."""
        out: dict[str, set[str]] = {}
        for transport, ids in (raw or {}).items():
            out[str(transport)] = {str(i) for i in (ids or [])}
        return out

    def role(self, principal: Principal) -> str:
        uid = str(principal.user_id)
        if uid in self.owners.get(principal.transport, set()):
            return ROLE_OWNER
        if uid in self.guests.get(principal.transport, set()):
            return ROLE_GUEST
        return ROLE_STRANGER

    def check_message(self, principal: Principal) -> Decision:
        """Gate for any interaction with the agent."""
        role = self.role(principal)
        if role == ROLE_STRANGER and not self.allow_strangers:
            return Decision(
                allowed=False,
                reason=(
                    f"unknown {principal.transport} user "
                    f"'{principal.user_id}' is not authorized"
                ),
            )
        return Decision(allowed=True)

    def check_tool(self, principal: Principal, tool_name: str) -> Decision:
        """Gate for tool execution. Tools never run for strangers."""
        role = self.role(principal)
        if role == ROLE_OWNER:
            return Decision(allowed=True)
        if role == ROLE_GUEST and not self.owner_only_tools:
            return Decision(allowed=True)
        return Decision(
            allowed=False,
            reason=(
                f"tool '{tool_name}' requires owner privileges "
                f"(caller role: {role})"
            ),
        )
