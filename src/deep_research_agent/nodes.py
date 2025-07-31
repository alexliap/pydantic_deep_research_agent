from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext

from .agents import (
    briefing_agent,
    clarifying_agent,
    report_writer,
    research_sub_agent,
    research_supervisor,
)
from .messages import ResearchTopic
from .prompts import compress_research_simple_human_message
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
    briefing: str | None = None
    acquired_results: list[str] | None = None

    async def run(
        self, ctx: GraphRunContext[ResearchState]
    ) -> Researcher | FinalReport:
        if self.briefing:
            supervisor_plan = await research_supervisor.run(
                self.briefing, message_history=ctx.state.supervisor_messages
            )

            ctx.state.supervisor_plan = supervisor_plan.output.plan

            ctx.state.supervisor_messages += supervisor_plan.new_messages()

        if self.acquired_results:
            supervisor_plan = await research_supervisor.run(
                self.acquired_results, message_history=ctx.state.supervisor_messages
            )

            ctx.state.supervisor_plan = supervisor_plan.output.plan

            ctx.state.supervisor_messages += supervisor_plan.new_messages()

        if supervisor_plan.output.proceed_to_final_report:
            return FinalReport()
        else:
            return Researcher(supervisor_plan.output.research_topics)


@dataclass
class Researcher(BaseNode[ResearchState, None, str]):
    research_topics: list[ResearchTopic]

    async def run(self, ctx: GraphRunContext[ResearchState]) -> Supervisor:
        async def agent_task(task: str):
            result = await research_sub_agent.run(task)
            return result.output

        research_sub_agents = [
            agent_task(topic.research_topic) for topic in self.research_topics
        ]

        multiple_tasks = await asyncio.gather(
            *research_sub_agents, return_exceptions=True
        )
        multiple_tasks = [
            result for result in multiple_tasks if not isinstance(result, BaseException)
        ]

        return Supervisor(
            acquired_results=[
                result
                for result in multiple_tasks
                if not isinstance(result, BaseException)
            ]
        )


@dataclass
class FinalReport(BaseNode[ResearchState, None, str]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> End[str]:
        final_report = await report_writer.run(
            compress_research_simple_human_message,
            message_history=ctx.state.supervisor_messages,
        )

        return End(final_report.output)
