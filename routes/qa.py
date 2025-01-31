from fastapi import APIRouter, HTTPException
import numpy as np
from models.schemas import QuestionRequest, SimilarQuestionResult
from services.embedding import embedding_service  # Explicit import
from services.database import pocketbase_service  # Explicit import

router = APIRouter(prefix="/qa", tags=["Q&A"])


def cosine_similarity(a: list, b: list) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@router.post("/search", response_model=list[SimilarQuestionResult])
async def search_similar_questions(request: QuestionRequest):
    try:
        # Generate query embedding
        query_embedding = embedding_service.get_embedding(request.question)

        # Get all stored questions
        records = pocketbase_service.get_all_questions()

        # Calculate similarities
        results = []
        for record in records:
            stored_embedding = record.get('embedding', [])
            if len(stored_embedding) != len(query_embedding):
                continue

            similarity = cosine_similarity(query_embedding, stored_embedding)
            results.append(SimilarQuestionResult(
                question=record.get('question', ''),
                similarity=similarity,
                id=record.get('id', '')
            ))

        # Return top 5 results
        return sorted(results, key=lambda x: x.similarity, reverse=True)[:5]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))