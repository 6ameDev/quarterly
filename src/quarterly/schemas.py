from pydantic import BaseModel


class IngestRequest(BaseModel):
    text: str
    metadata: dict[str, str] | None = None


class QuestionRequest(BaseModel):
    question: str
