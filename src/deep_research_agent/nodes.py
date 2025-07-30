from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext

from .agents import (
    briefing_agent,
    clarifying_agent,
    research_sub_agent,
    research_supervisor,
)
from .messages import ResearchTopic
from .state import ResearchState


@dataclass
class BeginResearch(BaseNode[ResearchState]):
    query: str

    async def run(
        self, ctx: GraphRunContext[ResearchState]
    ) -> WriteResearchBrief | End[str]:
        decision = await clarifying_agent.run(self.query)

        ctx.state.clarifying_agent_messages += decision.new_messages()

        if decision.output.need_clarification:
            return End(decision.output.question)
        else:
            ctx.state.original_prompt = self.query
            return WriteResearchBrief()


@dataclass
class WriteResearchBrief(BaseNode[ResearchState, None, str]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> Supervisor:
        briefing = await briefing_agent.run(ctx.state.original_prompt)

        ctx.state.briefing_agent_messages += briefing.new_messages()

        ctx.state.briefing = briefing.output.research_brief

        return Supervisor(briefing.output.research_brief)


@dataclass
class Supervisor(BaseNode[ResearchState]):
    briefing: str

    async def run(self, ctx: GraphRunContext[ResearchState]) -> Researcher:
        supervisor_plan = await research_supervisor.run(
            self.briefing, message_history=ctx.state.supervisor_messages
        )

        ctx.state.supervisor_messages += supervisor_plan.new_messages()

        return Researcher(supervisor_plan.output.research_topics)


@dataclass
class Researcher(BaseNode[ResearchState, None, str]):
    research_topics: list[ResearchTopic]

    async def run(self, ctx: GraphRunContext[ResearchState]) -> End[str]:
        async def agent_task(task: str):
            result = await research_sub_agent.run(task)
            return result.output

        research_sub_agents = [
            agent_task(topic.research_topic) for topic in self.research_topics
        ]

        multiple_tasks = await asyncio.gather(
            *research_sub_agents, return_exceptions=True
        )

        # TODO: add tools to researcher and complete the graph
        print(multiple_tasks)

        return End("")
