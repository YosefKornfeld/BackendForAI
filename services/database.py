import requests
import numpy as np
from typing import List, Dict, Tuple
from config import settings


class PocketBaseVectorSearch:
    def __init__(self):
        self.base_url = settings.pocketbase_url
        self._auth_token = None
        self._embedding_cache = None  # Stores (id, embedding) tuples
        self._authenticate()

    def _authenticate(self):
        """Admin authentication with email/password"""
        auth_url = f"{self.base_url}/api/admins/auth-with-password"
        response = requests.post(auth_url, json={
            "identity": settings.admin_email,
            "password": settings.admin_password
        })
        response.raise_for_status()
        self._auth_token = response.json()['token']

    def _get_all_embeddings(self) -> List[Tuple[str, list]]:
        """Fetch and cache all embeddings with their PocketBase IDs"""
        if not self._embedding_cache:
            url = f"{self.base_url}/api/collections/questions/records?fields=id,embedding"
            headers = {"Authorization": f"Bearer {self._auth_token}"}

            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                items = response.json()['items']
                self._embedding_cache = [
                    (item['id'], np.array(item['embedding']))
                    for item in items
                ]
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Failed to fetch embeddings: {str(e)}")

        return self._embedding_cache

    def find_nearest_ids(self, query_embedding: list, k: int = 5) -> List[str]:
        """Find top k similar embeddings and return their PocketBase IDs"""
        query_vec = np.array(query_embedding)
        embeddings = self._get_all_embeddings()

        similarities = []
        for pb_id, stored_vec in embeddings:
            norm = np.linalg.norm(stored_vec) * np.linalg.norm(query_vec)
            similarity = np.dot(stored_vec, query_vec) / norm if norm != 0 else 0
            similarities.append((pb_id, similarity))

        # Sort by similarity descending
        sorted_ids = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        return [pb_id for pb_id, _ in sorted_ids]

    def get_full_records(self, ids: List[str]) -> List[Dict]:
        """Fetch complete records by PocketBase IDs"""
        url = f"{self.base_url}/api/collections/questions/records"
        headers = {"Authorization": f"Bearer {self._auth_token}"}
        params = {"filter": f'id = "{",".join(ids)}"'}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()['items']
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch records: {str(e)}")


pb_vector_search = PocketBaseVectorSearch()