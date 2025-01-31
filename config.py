from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    pocketbase_url: str
    pocketbase_token: str
    embedding_model: str = "HeNLP/HeRo"

    class Config:
        env_file = ".env"


settings = Settings()