from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    surrealdb_url: str = "https://sudb.elgrab.li/rpc"
    surrealdb_namespace: str = "halacha_data" 
    surrealdb_database: str = "halacha" 
    surrealdb_username: str = "Ephraim"
    surrealdb_password: str = "Af!131044"
    surrealdb_table: str = "halacha_data"
    embedding_model: str = "HeNLP/HeRo"

    class Config:
        env_file = ".env"

settings = Settings()
