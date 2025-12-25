from pydantic_graph import BaseNode, End, FullStatePersistence, Graph

from deep_research_agent.nodes import (
    BeginResearch,
    FinalReport,
    Researcher,
    Supervisor,
    WriteResearchBrief,
)
from deep_research_agent.state import ResearchState


def get_graph_status(node: BaseNode):
    if isinstance(node, BeginResearch):
        return {"status": "Starting research ..."}
    elif isinstance(node, WriteResearchBrief):
        return {"status": "Preparing research brief ..."}
    elif isinstance(node, Supervisor):
        if node.acquired_results:
            return {"status": "Reflecting ..."}
        else:
            return {"status": "Creating plan of action ..."}
    elif isinstance(node, Researcher):
        return {"status": "Searching the web ..."}
    elif isinstance(node, FinalReport):
        return {"status": "Writing final report ..."}


def node_result(node: BaseNode, state: ResearchState):
    if isinstance(node, BeginResearch):
        return {"status": "begin_research", "content": state.verification}
    elif isinstance(node, WriteResearchBrief):
        return {"status": "write_research_brief", "content": state.research_brief}
    elif isinstance(node, Supervisor):
        return {"status": "supervisor", "content": state.supervisor_plan}
    elif isinstance(node, Researcher):
        return {
            "status": "researcher",
            "content": [topic.research_topic for topic in node.research_topics],
        }
    elif isinstance(node, FinalReport):
        return {"status": None, "content": None}


async def run_iter_graph(state: ResearchState):
    persistence = FullStatePersistence()
    graph = Graph(
        state_type=ResearchState,
        nodes=[BeginResearch, WriteResearchBrief, Supervisor, Researcher, FinalReport],
    )
    async with graph.iter(
        BeginResearch(
            "make a brief/short comparison between pixi and uv package managers"
        ),
        state=state,
        persistence=persistence,
    ) as run:
        node = run.next_node
        while not isinstance(node, End):
            current_node = node
            yield get_graph_status(current_node)
            node = await run.next(current_node)
            yield node_result(current_node, run.state)

        yield {"status": "final_report", "content": run.result.output}
