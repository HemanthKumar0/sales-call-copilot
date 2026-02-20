"""Storage layer for metadata and vector store management."""

import json
import logging
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config import Settings
from src.models import CallMetadata

logger = logging.getLogger(__name__)


class MetadataStore:
    """Manages call metadata persistence in a JSON file."""

    def __init__(self, metadata_file: str) -> None:
        """Initialize with path to calls_metadata.json."""
        self.metadata_file = Path(metadata_file)
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create the JSON file with empty array if it doesn't exist."""
        if not self.metadata_file.exists():
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            self.metadata_file.write_text("[]", encoding="utf-8")
            logger.info("Created metadata file: %s", self.metadata_file)

    def load_all(self) -> list[CallMetadata]:
        """Load all call metadata entries from JSON file."""
        data = json.loads(self.metadata_file.read_text(encoding="utf-8"))
        return [CallMetadata(**entry) for entry in data]

    def get_by_id(self, call_id: str) -> Optional[CallMetadata]:
        """Retrieve a single call metadata entry by call_id."""
        for entry in self.load_all():
            if entry.call_id == call_id:
                return entry
        return None

    def get_last(self) -> Optional[CallMetadata]:
        """Return the most recently ingested call metadata entry.

        Returns the entry with the lexicographically greatest date string,
        since ISO format ensures chronological ordering equals lexicographic ordering.
        """
        entries = self.load_all()
        if not entries:
            return None
        return max(entries, key=lambda e: e.date)

    def append(self, metadata: CallMetadata) -> None:
        """Append a new call metadata entry to the JSON file."""
        entries = self.load_all()
        entries.append(metadata)
        data = [entry.model_dump() for entry in entries]
        self.metadata_file.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        logger.info("Appended metadata for call: %s", metadata.call_id)


def get_vector_store(settings: Settings) -> Chroma:
    """Initialize and return a persistent Chroma vector store instance.

    Uses OpenAIEmbeddings with the configured embedding model and
    persists data to the configured directory.
    """
    embedding_function = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding_function,
        persist_directory=settings.persist_directory,
    )
