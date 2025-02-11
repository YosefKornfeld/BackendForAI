from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    surrealdb_url: str = os.getenv('SURREALDB_URL')
    surrealdb_namespace: str = "halacha_data"
    surrealdb_database: str = "halacha"
    surrealdb_username: str = os.getenv('SURREALDB_USERNAME')
    surrealdb_password: str = os.getenv('SURREALDB_PASSWORD')
    surrealdb_table: str = "halacha_data"
    embedding_model: str = os.getenv('EMBEDDING_MODEL')
    openai_api_key: str = os.getenv('OPENAI_API_KEY')
    serp_api_key: str = os.getenv('SERP_API_KEY')

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }

settings = Settings()
