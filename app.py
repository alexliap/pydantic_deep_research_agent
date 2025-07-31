import logging
from argparse import ArgumentParser
from contextlib import asynccontextmanager

import logfire
import uvicorn
from fastapi import FastAPI
from pydantic_graph import Graph

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup (startup)
    state = ResearchState()
    graph = Graph(
        nodes=(BeginResearch, WriteResearchBrief, Supervisor, Researcher, FinalReport)
    )

    app.state.state = state
    app.state.graph = graph

    yield

    # Teardown (shutdown)
    logging.info("Shutting down DeepResearchAPI...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"status": 200}


@app.post("/run_graph")
async def run_deep_research_graph(query: str):
    result = await app.state.graph.run(
        start_node=BeginResearch(query=query),
        state=app.state.state,
    )

    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="8000")
    parser.add_argument("--reload", default=True)

    args = parser.parse_args()

    uvicorn.run("app:app", host=args.host, port=int(args.port), reload=args.reload)
