from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import logfire
from ddgs import DDGS
from pydantic_graph import BaseNode, End, GraphRunContext

from deep_research_agent.agents import (
    briefing_agent,
    clarifying_agent,
    draft_summary_agent,
    report_writer,
    research_sub_agent,
    research_sub_agent_summarizer,
    research_supervisor,
)
from deep_research_agent.messages import ResearchTopic
from deep_research_agent.prompts import (
    compress_research_simple_human_message,
    research_round_prompt,
)
from deep_research_agent.state import ResearchState
from deep_research_agent.utils import remove_generated_refs, remove_md_headers


@dataclass
class BeginResearch(BaseNode[ResearchState]):
    query: str

    async def run(
        self, ctx: GraphRunContext[ResearchState]
    ) -> WriteResearchBrief | End[str]:
        decision = await clarifying_agent.run(self.query)

        ctx.state.clarifying_agent_messages += decision.new_messages()[
            0
        ].parts  # ty:ignore[unsupported-operator]

        if decision.output.need_clarification:
            return End(decision.output.question)
        else:
            ctx.state.original_prompt = self.query
            return WriteResearchBrief()


@dataclass
class WriteResearchBrief(BaseNode[ResearchState, None, str]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> Supervisor:
        briefing = await briefing_agent.run(ctx.state.original_prompt)

        ctx.state.briefing_agent_messages += briefing.new_messages()[
            0
        ].parts  # ty:ignore[unsupported-operator]

        ctx.state.research_brief = briefing.output.research_brief

        return Supervisor(research_brief=briefing.output.research_brief)


@dataclass
class Supervisor(BaseNode[ResearchState]):
    research_brief: str | None = None
    acquired_results: list[str] | None = None
    references: list[str] | None = None

    async def run(
        self, ctx: GraphRunContext[ResearchState]
    ) -> Researcher | FinalReport:
        if self.research_brief:
            ctx.state.current_research_iterations += 1

            logfire.info(
                f"Research Iteration: {ctx.state.current_research_iterations}/{ctx.state.max_research_iterations} ..."
            )

            supervisor_plan = await research_supervisor.run(self.research_brief)

            ctx.state.supervisor_plan = supervisor_plan.output.plan

        if self.acquired_results:
            ctx.state.current_research_iterations += 1

            logfire.info(
                f"Research Iteration: {ctx.state.current_research_iterations}/{ctx.state.max_research_iterations} ..."
            )

            if self.references:
                ctx.state.references += self.references

            prompt = research_round_prompt.format(
                running_summary=ctx.state.running_summary,
                results="\n\n".join(self.acquired_results),
            )

            supervisor_plan = await research_supervisor.run(prompt)

            updated_running_summary = await draft_summary_agent.run(prompt)

            # update the plan
            ctx.state.supervisor_plan = supervisor_plan.output.plan
            # update state running summary
            ctx.state.running_summary = updated_running_summary.output

        if supervisor_plan.output.proceed_to_final_report:
            return FinalReport()
        elif ctx.state.current_research_iterations == ctx.state.max_research_iterations:
            return FinalReport()
        else:
            return Researcher(research_topics=supervisor_plan.output.research_topics)


@dataclass
class Researcher(BaseNode[ResearchState, None, str]):
    research_topics: list[ResearchTopic]

    async def run(self, ctx: GraphRunContext[ResearchState]) -> Supervisor:
        async def agent_task(task: str):
            try:
                search_queries = await research_sub_agent.run(task)

                ddg_client = DDGS()

                search_results = []
                for query in search_queries.output.search_queries:
                    try:
                        output = ddg_client.text(query, max_results=5)
                        search_results.append(output)
                    except Exception as e:
                        msg = (
                            "Something went wrong when trying to gather agent's results"
                        )
                        logfire.error(f"{msg}: {e}")

                results = f"**{task}**: \n\n"
                refs = []
                for result in search_results:
                    for item in result:
                        refs.append(item["href"])
                        results += (
                            f"Title: {item['title'].strip()}"
                            + "\n"
                            + f"Content: {item['body'].strip()}"
                            + "\n\n"
                        )

                results_summary = await research_sub_agent_summarizer.run(
                    results.strip()
                )

                return results_summary.output, refs
            except Exception as e:
                msg = "Something went wrong when trying ot gather agent's result"
                logfire.error(f"{msg}: {e}")
                return f"No information gathered about: {task}.", []

        research_sub_tasks = [
            agent_task(topic.research_topic) for topic in self.research_topics
        ]

        multiple_tasks = await asyncio.gather(
            *research_sub_tasks, return_exceptions=True
        )

        findings = [result[0] for result in multiple_tasks]

        references = []
        for result in multiple_tasks:
            references += result[1]

        return Supervisor(acquired_results=findings, references=references)


@dataclass
class FinalReport(BaseNode[ResearchState, None, str]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> End[str]:
        final_report = await report_writer.run(
            compress_research_simple_human_message.format(
                running_summary=ctx.state.running_summary
            ),
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
