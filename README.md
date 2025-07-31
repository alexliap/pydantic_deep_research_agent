# Open Deep Research

<!-- ![deep_research_graph](https://github.com/alexliap/pydantic_deep_research_agent/blob/main/deep_research_graph.jpg) -->

<p align="center">
  <img src="https://github.com/alexliap/pydantic_deep_research_agent/blob/main/deep_research_graph.jpg" />
</p>


Deep research feature is one of the core tools in today's chatbot applications. This is a simple, configurable, fully open source deep research agent that works across many model providers, and search tools. This implementation is using Pydantic AI and is inspired by the [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) repository.

- Read Langchain's [blog](https://blog.langchain.com/open-deep-research/) for a high level overview.

## Quickstart

```shell
git clone https://github.com/alexliap/pydantic_deep_research_agent.git

cd pydantic_deep_research_agent

uv venv

uv sync
```

Currently inside the `src/deep_research_agent/agents.py` file all agents use OpenAI models, so an `OPENAI_API_KEY` environment variable is needed.

Also, `src/deep_research_agent/agents.py` can obviously be modified to accept other models ([Pydantic AI Model Providers](https://ai.pydantic.dev/models/)).

## Example

One available example with the running graph exists in `main.py`, where the query is:

 `Detailed report between python package managers pixi and uv.`

```shell
uv run main.py
```

## Server

In case you want to serve the Agent using fastapi

```shell
uv pip install -e ".[fastapi]"

uv run app.py
```

## Observability

The project uses [logfire](https://github.com/pydantic/logfire) for observability.
