from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DB_URL: str
    STORAGE_PATH: str = "data/raw"
    VECTOR_STORE_BACKEND: str = "faiss"
    FAISS_INDEX_PATH: str = "./faiss_index"
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()  # type: ignore