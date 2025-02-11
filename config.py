from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    surrealdb_url: str = "https://sudb.elgrab.li/rpc"
    surrealdb_namespace: str = "halacha_data"
    surrealdb_database: str = "halacha"
    surrealdb_username: str = "Ephraim"
    surrealdb_password: str = "Af!131044"
    surrealdb_table: str = "halacha_data"
    embedding_model: str = "HeNLP/HeRo"
    openai_api_key: str = "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"
    serp_api_key: str = "a7af41747605ae79916d2e03b231432d4d8e8c14d17bedf670856b60e0547575" 

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }

settings = Settings()
