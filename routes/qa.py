from fastapi import APIRouter, HTTPException
from models.schemas import QuestionRequest, SimilarQuestionResult
from services.embedding import embedding_service
from services.database import surreal_vector_search

router = APIRouter(prefix="/qa", tags=["Q&A"])

@router.post("/ask", response_model=list[SimilarQuestionResult])
async def ask_question(request: QuestionRequest):
    try:
        # Step 1: Generate the query embedding using HeRo (or your chosen model)
        query_embedding = embedding_service.get_embedding(request.question)
        
        # Step 2: Perform vector search in SurrealDB
        results = surreal_vector_search.find_nearest(query_embedding)
        
        # Step 3: Map the database result to the API schema.
        response = []
        for rec in results:
            response.append(SimilarQuestionResult(
                id=rec["id"],
                question=rec["question"],
                similarity=rec["similarity"]
            ))
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
