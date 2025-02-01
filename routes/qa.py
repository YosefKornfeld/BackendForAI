from fastapi import APIRouter, HTTPException
from models.schemas import QuestionRequest, SimilarQuestionResult
from services.embedding import embedding_service
from services.database import pb_vector_search

router = APIRouter(prefix="/qa", tags=["Q&A"])


@router.post("/search", response_model=list[SimilarQuestionResult])
async def search_questions(request: QuestionRequest):
    try:
        # Step 1: Generate embedding for the question
        embedding = embedding_service.get_embedding(request.question)

        # Step 2: Find nearest IDs
        top_ids = pb_vector_search.find_nearest_ids(embedding)

        # Step 3: Get full records
        records = pb_vector_search.get_full_records(top_ids)

        # Step 4: Format results maintaining order
        id_to_record = {record['id']: record for record in records}
        return [
            SimilarQuestionResult(
                question=id_to_record[pb_id]['Question'],  # Updated to 'Question'
                similarity=0,  # Not tracking actual score in this flow
                id=pb_id
            ) for pb_id in top_ids if pb_id in id_to_record
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))