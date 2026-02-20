"""Ingestion engine for reading, chunking, and storing sales call transcripts."""

import logging
import os
import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import Settings
from src.models import CallMetadata
from src.storage import MetadataStore
from src.utils import extract_title_from_filename, generate_call_id, get_current_date_iso

logger = logging.getLogger(__name__)

SPEAKER_PATTERN = re.compile(r"^(?:\[[\d:]+\]\s*)?([^:]+):\s*(.+)", re.MULTILINE)
TIMESTAMP_PATTERN = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")


class IngestionEngine:
    """Handles reading, chunking, and storing transcript files."""

    def __init__(self, vector_store: Chroma, metadata_store: MetadataStore, settings: Settings) -> None:
        """Initialize with vector store, metadata store, and settings."""
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.settings = settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def ingest_file(self, filepath: str) -> CallMetadata:
        """Ingest a single transcript file into the vector store.

        1. Validate file exists and is non-empty
        2. Read file contents
        3. Generate call_id
        4. Split into chunks
        5. Attach metadata to each chunk
        6. Add documents to Chroma
        7. Update calls_metadata.json
        8. Return the new CallMetadata
        """
        # Validate file exists
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Transcript file not found: {filepath}")

        # Read and validate non-empty
        content = open(filepath, "r", encoding="utf-8").read()
        if not content.strip():
            raise ValueError(f"Transcript file is empty: {filepath}")

        call_id = generate_call_id()
        filename = os.path.basename(filepath)
        chunks = self._split_transcript(content)

        # Build LangChain documents with metadata
        documents = []
        for idx, chunk_text in enumerate(chunks):
            metadata = {
                "call_id": call_id,
                "segment_id": f"{call_id}_seg_{idx}",
                "speaker": self._extract_speaker(chunk_text),
                "timestamp": self._extract_timestamp(chunk_text),
                "source": filename,
            }
            documents.append(Document(page_content=chunk_text, metadata=metadata))

        # Add to vector store
        self.vector_store.add_documents(documents)
        logger.info("Added %d segments for call %s to vector store", len(documents), call_id)

        # Update metadata store
        call_metadata = CallMetadata(
            call_id=call_id,
            filename=filename,
            date=get_current_date_iso(),
            title=extract_title_from_filename(filename),
        )
        self.metadata_store.append(call_metadata)

        return call_metadata

    def _split_transcript(self, text: str) -> list[str]:
        """Split transcript text into chunks using RecursiveCharacterTextSplitter."""
        return self.splitter.split_text(text)

    def _extract_speaker(self, chunk: str) -> str:
        """Extract the first speaker name from chunk text.

        Returns 'Unknown' if no speaker pattern is detected.
        """
        match = SPEAKER_PATTERN.search(chunk)
        if match:
            return match.group(1).strip()
        return "Unknown"

    def _extract_timestamp(self, chunk: str) -> Optional[str]:
        """Extract the first timestamp from chunk text.

        Returns None if no timestamp pattern is found.
        """
        match = TIMESTAMP_PATTERN.search(chunk)
        if match:
            return match.group(1)
        return None
