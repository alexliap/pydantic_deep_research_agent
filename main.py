import asyncio

import logfire
from pydantic_graph import Graph

from deep_research_agent.nodes import (
    BeginResearch,
    Researcher,
    Supervisor,
    WriteResearchBrief,
)
from deep_research_agent.state import ResearchState

logfire.configure()
logfire.instrument_pydantic_ai()


async def run_graph():
    state = ResearchState()
    graph = Graph(nodes=(BeginResearch, WriteResearchBrief, Supervisor, Researcher))
    result = await graph.run(
        start_node=BeginResearch(query="what are the 10 best restaurants in Athens?"),
        state=state,
    )

    return result.output


print(asyncio.run(run_graph()))
