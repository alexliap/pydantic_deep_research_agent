from pydantic import BaseModel, Field


class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
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
    general_plan: str
    """General layed out plan about the research."""
    research_topics: list[ResearchTopic]
    """List of research topics to be given to sub-agents."""
