# routes/qa.py
from fastapi import APIRouter, HTTPException
from models.schemas import QuestionRequest, QAResponse
from services.embedding import embedding_service
from services.database import surreal_vector_search
from services.gpt4o_mini import get_gpt4mini_answer

router = APIRouter()

@router.post("/qa/ask", response_model=QAResponse)
async def ask_question(request: QuestionRequest):
    print("Received question:", request.question)
    query_embedding = embedding_service.get_embedding(request.question)
    print("Generated embedding:", query_embedding)
    
    results = surreal_vector_search.find_nearest_alternative(query_embedding, k=5)
    print("Database results:", results)
    
    # If no results, fallback to GPT
    if not results:
        gpt_response = get_gpt4mini_answer(request.question)
        print("GPT fallback response:", gpt_response)
        return gpt_response

    # Use the correct lowercase keys
    main_answer = results[0].get("answer", "No answer found.")
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