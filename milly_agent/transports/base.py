"""base.py — abstract transport interface."""

from abc import ABC, abstractmethod

from milly_agent.agent import Agent
from milly_agent.authz import Principal


class Transport(ABC):
    """A channel that delivers user messages to the agent and replies back.

    Subclasses set ``name`` (used as the Principal.transport value, which is
    what authz owner/guest lists are keyed on) and implement ``run()``, which
    blocks for the lifetime of the process.
    """

    name: str = "base"

    def __init__(self, agent: Agent, config: dict | None = None):
        self.agent = agent
        self.config = config or {}
        self.transport_config: dict = (
            (self.config.get("transports") or {}).get(self.name) or {}
        )

    def principal(self, user_id: str | int, display_name: str = "") -> Principal:
        return Principal(
            transport=self.name,
            user_id=str(user_id),
            display_name=display_name,
        )

    @abstractmethod
    def run(self) -> None:
        """Start the transport loop. Blocks until the transport shuts down."""
