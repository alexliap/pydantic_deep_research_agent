import asyncio

import logfire
from pydantic_graph import BaseNode, Graph

from deep_research_agent.nodes import (
    BeginResearch,
    FinalReport,
    Researcher,
    Supervisor,
    WriteResearchBrief,
)
from deep_research_agent.state import ResearchState

logfire.configure()
logfire.instrument_pydantic_ai()


def generate_graph_image(graph: Graph, start_node: type[BaseNode], img_name: str):
    import io

    import PIL.Image as Image

    image_bytes = graph.mermaid_image(start=start_node)
    Image.open(io.BytesIO(image_bytes)).save(img_name)


async def run_graph():
    state = ResearchState()
    graph = Graph(
        nodes=(BeginResearch, WriteResearchBrief, Supervisor, Researcher, FinalReport)
    )

    generate_graph_image(graph, BeginResearch, "deep_research_graph.jpg")

    result = await graph.run(
        start_node=BeginResearch(
            query="Detailed report between python package managers pixi and uv."
        ),
        state=state,
    )

    return result.output


print(asyncio.run(run_graph()))
