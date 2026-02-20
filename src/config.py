"""Centralized configuration for Sales Call Copilot."""

import logging
import sys

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables and defaults."""

    model_config = ConfigDict(env_file=".env")

    openai_api_key: str
    chunk_size: int = 600
    chunk_overlap: int = 100
    collection_name: str = "sales_calls"
    persist_directory: str = "./chroma_db"
    llm_model: str = "gpt-5-mini"
    llm_temperature: float = 0.0
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5
    metadata_file: str = "calls_metadata.json"
    data_directory: str = "./data"


def get_settings() -> Settings:
    """Load and validate settings. Raises if OPENAI_API_KEY is missing."""
    try:
        return Settings()
    except Exception as e:
        if "openai_api_key" in str(e).lower():
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment."
            ) from e
        raise


def setup_logging() -> None:
    """Configure the root logger with INFO level and a stderr StreamHandler."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return  # Already configured, avoid duplicate handlers

    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
