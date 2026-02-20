"""Prompt templates and formatting utilities for Sales Call Copilot."""

from langchain_core.documents import Document


SUMMARIZATION_PROMPT = """You are a sales call analyst. Summarize the following sales call transcript segments clearly and concisely. Highlight key topics discussed, action items, and any important decisions made.

Context:
{context}

Provide a structured summary. At the end of your response, you MUST include a Sources section in exactly this format:

Sources:
{sources}"""

QA_PROMPT = """You are a sales call analyst. Answer the following question using ONLY the provided context. Do not use any outside knowledge. If the context does not contain enough information to answer the question, respond with: "I don't have enough information in the provided transcripts to answer this question."

Context:
{context}

Question: {question}

Answer the question based solely on the context above. At the end of your response, you MUST include a Sources section in exactly this format:

Sources:
{sources}"""

SPECIFIC_QUERY_PROMPT = """You are a sales call analyst. Extract and list ONLY the content from the provided context that matches the following query. Do not add any information that is not present in the context. If no matching content is found, respond with: "I don't have enough information in the provided transcripts to answer this question."

Context:
{context}

Query: {question}

Extract only the relevant content that matches the query. At the end of your response, you MUST include a Sources section in exactly this format:

Sources:
{sources}"""

NO_CONTEXT_RESPONSE = "I don't have enough information in the provided transcripts to answer this question."


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a context string for the LLM prompt."""
    if not documents:
        return ""
    parts = []
    for doc in documents:
        meta = doc.metadata
        call_id = meta.get("call_id", "unknown")
        segment_id = meta.get("segment_id", "unknown")
        speaker = meta.get("speaker", "Unknown")
        timestamp = meta.get("timestamp", "")
        header = f"[Call {call_id}, Segment {segment_id}]"
        if speaker and speaker != "Unknown":
            header += f" {speaker}"
        if timestamp:
            header += f" ({timestamp})"
        parts.append(f"{header}:\n{doc.page_content}")
    return "\n\n".join(parts)


def format_sources(documents: list[Document]) -> str:
    """Format document metadata into the required source citation format."""
    if not documents:
        return ""
    lines = []
    for doc in documents:
        meta = doc.metadata
        call_id = meta.get("call_id", "unknown")
        segment_id = meta.get("segment_id", "unknown")
        snippet = doc.page_content[:80]
        lines.append(f'- Call {call_id}, Segment {segment_id}: "{snippet}"')
    return "\n".join(lines)
