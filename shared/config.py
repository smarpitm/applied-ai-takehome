"""Environment-driven settings for Problem 1 (RAG) and Problem 2 (LLM-as-Judge)."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelEndpointConfig(BaseModel):
    """Decoupled model routing config for a generator or judge role."""

    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    fallback_provider: str = "ollama"
    fallback_model: str = "llama3.1:latest"


class Settings(BaseSettings):
    """Application settings loaded from environment variables and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Cloud API keys (never logged or committed)
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # Local Ollama fallback
    enable_local_fallback: bool = True
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_p1_fallback_model: str = "llama3.1:latest"
    ollama_p2_generator_fallback_model: str = "llama3.1:latest"
    ollama_p2_judge_fallback_model: str = "qwen2.5:3b"

    # Problem 1 — RAG settings
    p1_vector_store: str = "lancedb"
    lancedb_path: Path = Path("./data/lancedb")
    lancedb_table: str = "rag_chunks"
    qdrant_url: str | None = "http://localhost:6333"
    qdrant_path: Path = Path("./data/qdrant")
    qdrant_collection: str = "rag_chunks"

    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    top_k: int = 5
    min_similarity_score: float = 0.665

    p1_llm_provider: str = "groq"
    p1_llm_model: str = "llama-3.1-8b-instant"
    p1_llm_temperature: float = 0.1
    p1_judge_provider: str = "gemini"
    p1_judge_model: str = "gemini-2.5-flash"

    # Problem 2 — LLM-as-Judge settings (decoupled generator vs judge)
    p2_generator_provider: str = "groq"
    p2_generator_model: str = "llama-3.3-70b-versatile"
    p2_generator_temperature: float = 0.2
    p2_judge_provider: str = "gemini"
    p2_judge_model: str = "gemini-2.5-flash"
    p2_judge_temperature: float = 0.0
    p2_pass_threshold: float = 3.5

    # Logging
    log_dir: Path = Path("./results/logs")
    log_level: str = "INFO"

    def get_api_key(self, provider: str) -> str | None:
        """Return plain API key string for a provider if configured and non-empty."""
        mapping = {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "openrouter": self.openrouter_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }
        secret = mapping.get(provider.lower())
        if secret is None:
            return None
        value = secret.get_secret_value().strip()
        return value or None

    def p1_generator_config(self) -> ModelEndpointConfig:
        """Return Problem 1 generation endpoint config."""
        return ModelEndpointConfig(
            provider=self.p1_llm_provider,
            model=self.p1_llm_model,
            temperature=self.p1_llm_temperature,
            max_tokens=768,
            fallback_provider="ollama",
            fallback_model=self.ollama_p1_fallback_model,
        )

    def p1_judge_config(self) -> ModelEndpointConfig:
        """Return Problem 1 answer-evaluation judge config (separate model)."""
        return ModelEndpointConfig(
            provider=self.p1_judge_provider,
            model=self.p1_judge_model,
            temperature=0.0,
            max_tokens=512,
            fallback_provider="ollama",
            fallback_model=self.ollama_p2_judge_fallback_model,
        )

    def p2_generator_config(self) -> ModelEndpointConfig:
        """Return Problem 2 generator endpoint config."""
        return ModelEndpointConfig(
            provider=self.p2_generator_provider,
            model=self.p2_generator_model,
            temperature=self.p2_generator_temperature,
            max_tokens=1024,
            fallback_provider="ollama",
            fallback_model=self.ollama_p2_generator_fallback_model,
        )

    def p2_judge_config(self) -> ModelEndpointConfig:
        """Return Problem 2 judge endpoint config (temperature 0 by default)."""
        return ModelEndpointConfig(
            provider=self.p2_judge_provider,
            model=self.p2_judge_model,
            temperature=self.p2_judge_temperature,
            max_tokens=2048,
            fallback_provider="ollama",
            fallback_model=self.ollama_p2_judge_fallback_model,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
