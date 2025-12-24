from dataclasses import dataclass, field

from pydantic_ai.messages import SystemPromptPart, UserPromptPart


@dataclass
class ResearchState:
    original_prompt: str | None = None
    research_brief: str | None = None
    supervisor_plan: str | None = None

    running_summary: str = field(default_factory=str)
    references: list[str] = field(default_factory=list)

    clarifying_agent_messages: list[SystemPromptPart | UserPromptPart] = field(
        default_factory=list
    )
    briefing_agent_messages: list[SystemPromptPart | UserPromptPart] = field(
        default_factory=list
    )
    supervisor_messages: list[SystemPromptPart | UserPromptPart] = field(
        default_factory=list
    )
