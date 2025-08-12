from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage


@dataclass
class ResearchState:
    original_prompt: str | None = None
    briefing: str | None = None
    supervisor_plan: str | None = None

    references: list[str] = field(default_factory=list)

    clarifying_agent_messages: list[ModelMessage] = field(default_factory=list)
    briefing_agent_messages: list[ModelMessage] = field(default_factory=list)
    supervisor_messages: list[ModelMessage] = field(default_factory=list)
