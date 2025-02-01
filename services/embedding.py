from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from config import settings

class EmbeddingService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        self.model = AutoModel.from_pretrained(settings.embedding_model)
        self.model.eval()  # Set model to evaluation mode

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
        # Mean pooling on token embeddings to get a fixed-size vector
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        if isinstance(embedding, float):
            return [embedding]
        return embedding

embedding_service = EmbeddingService()
