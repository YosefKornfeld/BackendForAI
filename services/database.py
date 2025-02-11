from typing import List, Dict
from config import settings
from surrealdb import Surreal
import json

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Simple pythonic cosine similarity if you still need it internally.
    Not used by the alternative SurrealDB-based method below.
    """
    from math import sqrt
    dot = sum(a*b for a, b in zip(vec1, vec2))
    mag1 = sqrt(sum(a*a for a in vec1))
    mag2 = sqrt(sum(b*b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

class SurrealDBVectorSearch:
    def __init__(self):
        self.client = Surreal(settings.surrealdb_url)
        # Authenticate and select the namespace and database
        self.client.signin({
            "username": settings.surrealdb_username,
            "password": settings.surrealdb_password
        })
        self.client.use(namespace=settings.surrealdb_namespace, database=settings.surrealdb_database)

    def find_nearest_bruteforce(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Brute force search:
         - Only fetches records with a 768-length embedding
         - Computes cosine similarity in Python
         - Returns top k results
        """
        query = f"""
        SELECT id, Question, Answer, embedding
        FROM {settings.surrealdb_table}
        WHERE array::len(embedding) = 768;
        """
        result = self.client.query(query)
        
        # SurrealDB often returns [[...]]; unwrap if necessary
        if result and isinstance(result[0], list):
            records = result[0]
        else:
            records = result

        # Compute cosine similarity in Python
        results = []
        for rec in records:
            emb = rec.get("embedding")
            if not emb or len(emb) != 768:
                continue
            sim = cosine_similarity(query_embedding, emb)
            rec["similarity"] = sim
            results.append(rec)

        # Sort by similarity descending and limit
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def find_nearest(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Original method using SurrealDB's vector search operator <|k,l|>.
        Requires SurrealDB 2.0+ plus an HNSW index definition.
        """
        query = f"""
        SELECT id, Question, Answer, vector::distance::knn() AS distance
        FROM {settings.surrealdb_table}
        WHERE embedding <|{k},40|> $query_embedding
        ORDER BY distance ASC
        LIMIT {k};
        """
        result = self.client.query(query, {"query_embedding": query_embedding})
        if result and isinstance(result[0], list):
            records = result[0]
        else:
            records = result

        output = []
        for rec in records:
            output.append({
                "id": rec.get("id"),
                "question": rec.get("Question", ""),
                "answer": rec.get("Answer", ""),
                "similarity": rec.get("distance")
            })
        return output

    def find_nearest_alternative(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Uses SurrealDB's built-in cosine similarity function.
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
        print("Running query:", query)
        print("Query embedding dimension:", len(query_embedding))

        result = self.client.query(query, {"query_embedding": query_embedding})
        print("Raw query result:", result)

        if result and isinstance(result[0], list):
            records = result[0]
        else:
            records = result

        output = []
        for rec in records:
            # If rec is a string, attempt to parse JSON (just in case the client returns strings)
            if isinstance(rec, str):
                try:
                    rec = json.loads(rec)
                except Exception as e:
                    print("Failed to parse record:", rec, e)
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
