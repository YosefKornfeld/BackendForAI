from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

class SimilarQuestionResult(BaseModel):
    question: str
    similarity: float
    id: str