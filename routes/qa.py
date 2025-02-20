# routes/qa.py
# routes/qa.py
from fastapi import APIRouter, HTTPException
from models.schemas import QuestionRequest, QAResponse
from models.schemas import QuestionRequest, QAResponse
from services.embedding import embedding_service
from services.database import surreal_vector_search
from services.gpt4o_mini import get_gpt4mini_answer
import json

router = APIRouter()

@router.post("/qa/ask", response_model=QAResponse)
async def ask_question(request: QuestionRequest):
    print("Received question:", request.question)
    query_embedding = embedding_service.get_embedding(request.question)
    print("Generated embedding:", query_embedding)
    
    results = surreal_vector_search.find_nearest_hnsw(query_embedding, k=10)
    print("Database results:", results)
    
    qa_list = [
        {
            "question": rec.get("question", ""),
            "answer": rec.get("answer", "")
        }
        for rec in results
    ]
    qa_pairs = json.dumps(qa_list)  # Convert qa_list to JSON string
    gpt_response = get_gpt4mini_answer(request.question, qa_pairs)
    print("GPT fallback response:", gpt_response)

    # Use the correct lowercase keys
    main_answer = gpt_response.get("answer", "No answer found.")
    qa_list = [
        {
            "question": rec.get("question", ""),
            "answer": rec.get("answer", "")
        }
        for rec in results
    ]
    
    response_data = {"answer": main_answer, "qa_list": qa_list}
    print("Returning response:", response_data)
    return response_data