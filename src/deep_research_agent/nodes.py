from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import logfire
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits
from pydantic_graph import BaseNode, End, GraphRunContext

from .agents import (
    briefing_agent,
    clarifying_agent,
    report_writer,
    research_sub_agent,
    research_supervisor,
)
from .messages import ResearcherOutput, ResearchTopic
from .prompts import compress_research_simple_human_message
from .state import ResearchState
from .utils import remove_generated_refs, remove_md_headers


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

        return Supervisor(briefing=briefing.output.research_brief)


@dataclass
class Supervisor(BaseNode[ResearchState]):
    briefing: str | None = None
    acquired_results: list[str] | None = None
    references: list[str] | None = None

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
            if self.references:
                ctx.state.references += self.references

            supervisor_plan = await research_supervisor.run(
                self.acquired_results, message_history=ctx.state.supervisor_messages
            )

            ctx.state.supervisor_plan = supervisor_plan.output.plan

            ctx.state.supervisor_messages += supervisor_plan.new_messages()

        if supervisor_plan.output.proceed_to_final_report:
            return FinalReport()
        else:
            return Researcher(research_topics=supervisor_plan.output.research_topics)


@dataclass
class Researcher(BaseNode[ResearchState, None, str]):
    research_topics: list[ResearchTopic]

    async def run(self, ctx: GraphRunContext[ResearchState]) -> Supervisor:
        async def agent_task(task: str):
            try:
                # the request limit parameters is supposedly limiting the amount of time the search tool
                # is going to be used, but up to now it does not
                result = await research_sub_agent.run(
                    task, usage_limits=UsageLimits(request_limit=10)
                )

                result = result.output

            except UsageLimitExceeded:
                logfire.info("Maximum number of tool calls was reached.")
                result = ResearcherOutput(
                    content=f"No information gathered about {task}.", references=[""]
                )
            except Exception as e:
                msg = "Something went wrong when trying ot gather agent's result"
                logfire.error(f"{msg}: {e}")
                result = ResearcherOutput(
                    content=f"No information gathered about {task}.", references=[""]
                )

            return result

        research_sub_tasks = [
            agent_task(topic.research_topic) for topic in self.research_topics
        ]

        multiple_tasks = await asyncio.gather(
            *research_sub_tasks, return_exceptions=True
        )

        findings = [result.content for result in multiple_tasks]

        references = []
        for result in multiple_tasks:
            references += result.references

        return Supervisor(acquired_results=findings, references=references)


@dataclass
class FinalReport(BaseNode[ResearchState, None, str]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> End[str]:
        final_report = await report_writer.run(
            compress_research_simple_human_message,
            message_history=ctx.state.supervisor_messages,
        )
        # remove markdown headers (```markdown & ```) if they exist
        report = remove_md_headers(final_report.output)
        # remove LLM generated references
        report = remove_generated_refs(report)

        if ctx.state.references:
            references = self._filter_refs(ctx.state.references)

        references_str = "- " + "\n- ".join(references)
        report_structure = report + f"\n\n## References:\n\n{references_str}"

        return End(report_structure)

    @staticmethod
    def _filter_refs(references: list[str]):
        filtered_references = list(set(references))

        result = []
        error_msg = "Not valid URL"
        for ref in filtered_references:
            if ref != "":
                try:
                    urlopen(ref, timeout=30)
                except HTTPError as e:
                    logfire.warning(f"{error_msg}: {e}")
                except URLError as e:
                    logfire.warning(f"{error_msg}: {e}")
                except Exception as e:
                    logfire.warning(f"{error_msg}: {e}")
                else:
                    result.append(ref)

        return result
