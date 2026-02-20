"""Retriever module for semantic search and LLM-powered responses."""

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import Settings
from src.prompts import (
    NO_CONTEXT_RESPONSE,
    SUMMARIZATION_PROMPT,
    format_context,
    format_sources,
)

logger = logging.getLogger(__name__)


class SalesCallRetriever:
    """Semantic search with metadata filtering and LLM response generation."""

    def __init__(self, vector_store: Chroma, settings: Settings) -> None:
        """Initialize with vector store and settings."""
        self.vector_store = vector_store
        self.settings = settings
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
        )

    def retrieve(
        self,
        query: str,
        call_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> list[Document]:
        """Retrieve top-k relevant segments with optional metadata filters.

        Returns LangChain Document objects with metadata.
        """
        filter_dict: Optional[dict] = None
        if call_id:
            filter_dict = {"call_id": call_id}

        results = self.vector_store.similarity_search(
            query,
            k=self.settings.top_k,
            filter=filter_dict,
        )

        logger.info("Retrieved %d segments for query (call_id=%s)", len(results), call_id)

        if date_from or date_to:
            results = self._filter_by_date(results, date_from, date_to)

        return results

    def query_with_llm(
        self,
        query: str,
        prompt_template: str,
        call_id: Optional[str] = None,
    ) -> str:
        """Full RAG pipeline: retrieve context, build prompt, call LLM, return response."""
        docs = self.retrieve(query, call_id=call_id)

        if not docs:
            return NO_CONTEXT_RESPONSE

        context = format_context(docs)
        sources = format_sources(docs)

        prompt = prompt_template.format(
            context=context,
            sources=sources,
            question=query,
        )

        try:
            response = self.llm.invoke(prompt)
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise
        return response.content

    def summarize_call(self, call_id: str) -> str:
        """Retrieve all segments for a call and generate a summary."""
        docs = self.retrieve("summarize this call", call_id=call_id)

        if not docs:
            return NO_CONTEXT_RESPONSE

        context = format_context(docs)
        sources = format_sources(docs)

        prompt = SUMMARIZATION_PROMPT.format(
            context=context,
            sources=sources,
        )

        try:
            response = self.llm.invoke(prompt)
        except Exception as exc:
            logger.error("OpenAI API call failed during summarization: %s", exc)
            raise
        return response.content

    def _filter_by_date(
        self,
        documents: list[Document],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> list[Document]:
        """Filter documents by date range using metadata."""
        filtered = []
        for doc in documents:
            doc_date = doc.metadata.get("date", "")
            if date_from and doc_date < date_from:
                continue
            if date_to and doc_date > date_to:
                continue
            filtered.append(doc)
        return filtered
