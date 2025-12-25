from dataclasses import dataclass, field

from pydantic_ai.messages import SystemPromptPart, UserPromptPart


@dataclass
class ResearchState:
    original_prompt: str = field(default_factory=str)
    # BeginResearch node
    verification: str = field(default_factory=str)
    # WriteResearchBrief node
    research_brief: str = field(default_factory=str)
    # Supervisor node
    supervisor_plan: str = field(default_factory=str)
    running_summary: str = field(default_factory=str)
    references: list[str] = field(default_factory=list)

    current_research_iterations: int = 0
    max_research_iterations: int = 5

    clarifying_agent_messages: list[SystemPromptPart | UserPromptPart] = field(
        default_factory=list
    )
    briefing_agent_messages: list[SystemPromptPart | UserPromptPart] = field(
        default_factory=list
    )
    supervisor_messages: list[SystemPromptPart | UserPromptPart] = field(
        default_factory=list
    )
