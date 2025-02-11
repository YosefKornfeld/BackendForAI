# models/schemas.py
from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    question: str

class QAPair(BaseModel):
    question: str
    answer: str

class QAResponse(BaseModel):
    answer: str        # The main answer (for example, from the top result)
    qa_list: List[QAPair]
