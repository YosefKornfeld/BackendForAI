from typing import List, Dict
from typing import List, Dict
from config import settings
from surrealdb import Surreal
import json

class SurrealDBVectorSearch:
    def __init__(self):
        self.client = Surreal(settings.surrealdb_url)
        # Authenticate and select the namespace and database
        self.client.signin({
            "username": settings.surrealdb_username,
            "password": settings.surrealdb_password
        })
        self.client.use(namespace=settings.surrealdb_namespace, database=settings.surrealdb_database)

    def find_nearest_hnsw(self, query_embedding: List[float], k: int = 10) -> List[Dict]:
        """
        Uses SurrealDB's HNSW index via the vector::distance::knn() operator to perform vector search.
        Filters rows to only those with a 768-dim embedding.
        """
        query = f"""
        SELECT
            id,
            Question,
            Answer,
            vector::distance::knn() AS distance
        FROM {settings.surrealdb_table}
        WHERE array::len(embedding) = 768 AND embedding <|{k},40|> $query_embedding
        ORDER BY distance ASC
        LIMIT {k};
        """
        result = self.client.query(query, {"query_embedding": query_embedding})
        records = result[0] if result and isinstance(result[0], list) else result

        output = []
        for rec in records:
            if isinstance(rec, str):
                try:
                    rec = json.loads(rec)
                except Exception:
                    continue
            if not isinstance(rec, dict):
                continue
            output.append({
                "id": rec.get("id"),
                "question": rec.get("Question", ""),
                "answer": rec.get("Answer", ""),
                "similarity": rec.get("distance")
            })
        return output
    
    def find_nearest_alternative(self, query_embedding: List[float], k: int = 10) -> List[Dict]:
        """
        Uses SurrealDB's built-in cosine similarity function to perform vector search.
        Filters rows to only those with a 768-dim embedding.
        """
        query = f"""
        SELECT
            id,
            Question,
            Answer,
            vector::similarity::cosine(embedding, $query_embedding) AS similarity
        FROM {settings.surrealdb_table}
        WHERE array::len(embedding) = 768
        ORDER BY similarity DESC
        LIMIT {k};
        """
        result = self.client.query(query, {"query_embedding": query_embedding})
        records = result[0] if result and isinstance(result[0], list) else result

        output = []
        for rec in records:
            if isinstance(rec, str):
                try:
                    rec = json.loads(rec)
                except Exception:
                    continue
            if not isinstance(rec, dict):
                continue
            output.append({
                "id": rec.get("id"),
                "question": rec.get("Question", ""),
                "answer": rec.get("Answer", ""),
                "similarity": rec.get("similarity")
            })
        return output

# Instantiate the updated service
surreal_vector_search = SurrealDBVectorSearch()
