import numpy as np
from typing import List, Dict
from config import settings
from surrealdb import Surreal

class SurrealDBVectorSearch:
    def __init__(self):
        self.client = Surreal(settings.surrealdb_url)
        # Authenticate and select the namespace and database
        self.client.signin({"user": settings.surrealdb_username, "pass": settings.surrealdb_password})
        self.client.use(namespace=settings.surrealdb_namespace, database=settings.surrealdb_database)

    def find_nearest(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Use SurrealDB's vector search to retrieve the k nearest halacha questions.
        The query uses the HNSW vector operator and the built-in knn function.
        (Note: Lower distance means more similar.)
        """
        # The query uses a parameterized vector for searching.
        query = f"""
        SELECT id, Question, vector::distance::knn() AS distance
        FROM {settings.surrealdb_table}
        WHERE embedding <|HNSW|> $query_embedding
        ORDER BY distance ASC
        LIMIT {k};
        """
        result = self.client.query(query, {"query_embedding": query_embedding})
        # Depending on the client’s response format, we normalize the result.
        # (Often, the result is a list where the first element contains the records.)
        if result and isinstance(result[0], list):
            records = result[0]
        else:
            records = result

        # Map the database result to a list of dictionaries.
        output = []
        for rec in records:
            output.append({
                "id": rec.get("id"),
                "question": rec.get("Question", ""),
                "similarity": rec.get("distance")  # distance (lower means more similar)
            })
        return output

# Instantiate the service to be used in our routes.
surreal_vector_search = SurrealDBVectorSearch()
