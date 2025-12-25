from pydantic import BaseModel, Field, model_validator


class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope.",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )


class ResearchQuestion(BaseModel):
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class ResearchTopic(BaseModel, use_attribute_docstrings=True):
    research_topic: str
    """The research topic the """


class ResearchList(BaseModel, use_attribute_docstrings=True):
    plan: str = Field(
        description="General layed out plan about the research.", default_factory=str
    )
    # """General layed out plan about the research."""
    research_topics: list[ResearchTopic] | None
    """List of research topics to be given to sub-agents."""
    proceed_to_final_report: bool
    """If the information is enough proceed to final report generation."""

    @model_validator(mode="after")
    def _check_response_validity(self):
        if self.research_topics is None and not self.proceed_to_final_report:
            raise ValueError(
                "research_topics can only be empty if proceed_to_final_report is True."
            )

        return self

    @model_validator(mode="after")
    def _limit_topics(self):
        if self.research_topics is not None:
            if len(self.research_topics) > 10:
                self.research_topics = self.research_topics[:10]

        return self


# class ResearcherOutput(BaseModel, use_attribute_docstrings=True):
#     content: str
#     """Research findings."""
#     references: list[str]
#     """URL links that were visited during the agent's research."""


class ResearcherOutput(BaseModel, use_attribute_docstrings=True):
    search_queries: list[str]
    """Queries that are going to be used for web search."""
