import requests
from typing import List, Dict
from config import settings


class PocketBaseService:
    def __init__(self):
        self.base_url = settings.pocketbase_url
        self.headers = {
            "Authorization": f"Bearer {settings.pocketbase_token}",
            "Content-Type": "application/json"
        }

    def get_all_questions(self) -> List[Dict]:
        url = f"{self.base_url}/api/collections/questions/records"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json().get('items', [])


pocketbase_service = PocketBaseService()