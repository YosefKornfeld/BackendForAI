import asyncio
from services.embedding import embedding_service
from services.database import pb_vector_search
from routes.qa import search_questions
from models.schemas import QuestionRequest


async def test_search():
    # Create a test question
    test_question = "Can I use electricity on Shabbat?"

    # Generate embedding
    embedding = embedding_service.get_embedding(test_question)
    print(f"Generated embedding: {embedding[:5]}...")  # Show first 5 dimensions

    # Find nearest IDs
    top_ids = pb_vector_search.find_nearest_ids(embedding)
    print(f"Top matching IDs: {top_ids}")

    # Fetch full records
    records = pb_vector_search.get_full_records(top_ids)
    print("Matching records:")
    for record in records:
        print(f"ID: {record['id']}, Question: {record['Question']}")

    # Test the full endpoint
    request = QuestionRequest(question=test_question)
    results = await search_questions(request)
    print("\nAPI Response:")
    for result in results:
        print(f"ID: {result.id}, Question: {result.question}")


if __name__ == "__main__":
    asyncio.run(test_search())