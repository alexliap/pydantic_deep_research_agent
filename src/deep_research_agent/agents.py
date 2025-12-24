from dotenv import load_dotenv
from pydantic_ai import Agent

# from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.models.openai import OpenAIChatModelSettings

from deep_research_agent.messages import (
    ClarifyWithUser,
    ResearcherOutput,
    ResearchList,
    ResearchQuestion,
)
from deep_research_agent.prompts import (
    clarify_with_user_instructions,
    compress_research_system_prompt,
    draft_summary_prompt,
    lead_researcher_prompt,
    research_summarizer_system_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from deep_research_agent.utils import get_date

load_dotenv()

OPENAI_MODEL_BIG = "openai:gpt-5"
OPENAI_MODEL_SMALL = "openai:gpt-4.1-nano"
OPENAI_MODEL_REASON = "openai:o4-mini"

clarifying_agent = Agent(
    OPENAI_MODEL_BIG,
    output_type=ClarifyWithUser,
    system_prompt=clarify_with_user_instructions.format(date=get_date()),
    model_settings=OpenAIChatModelSettings(openai_service_tier="flex"),
)

briefing_agent = Agent(
    OPENAI_MODEL_BIG,
    output_type=ResearchQuestion,
    system_prompt=transform_messages_into_research_topic_prompt.format(date=get_date()),
    model_settings=OpenAIChatModelSettings(openai_service_tier="flex"),
)

research_supervisor = Agent(
    OPENAI_MODEL_REASON,
    output_type=ResearchList,
    system_prompt=lead_researcher_prompt.format(date=get_date()),
    model_settings=OpenAIChatModelSettings(
        openai_reasoning_effort="high", openai_service_tier="flex"
    ),
)

research_sub_agent = Agent(
    OPENAI_MODEL_SMALL,
    output_type=ResearcherOutput,
    system_prompt=research_system_prompt.format(date=get_date()),
    model_settings=OpenAIChatModelSettings(openai_service_tier="default", timeout=120),
)

research_sub_agent_summarizer = Agent(
    OPENAI_MODEL_SMALL,
    output_type=str,
    system_prompt=research_summarizer_system_prompt.format(date=get_date()),
    model_settings=OpenAIChatModelSettings(openai_service_tier="default", timeout=120),
)

report_writer = Agent(
    OPENAI_MODEL_REASON,
    output_type=str,
    system_prompt=compress_research_system_prompt.format(date=get_date()),
    model_settings=OpenAIChatModelSettings(
        openai_reasoning_effort="high", openai_service_tier="flex"
    ),
)

draft_summary_agent = Agent(
    OPENAI_MODEL_REASON,
    output_type=str,
    system_prompt=draft_summary_prompt.format(date=get_date()),
    model_settings=OpenAIChatModelSettings(
        openai_reasoning_effort="high", openai_service_tier="flex"
    ),
)
