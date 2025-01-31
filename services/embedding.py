from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from config import settings


class EmbeddingService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        self.model = AutoModel.from_pretrained(settings.embedding_model)

    def get_embedding(self, text: str) -> List[float]:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()


embedding_service = EmbeddingService()