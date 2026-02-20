"""Shared helper functions for Sales Call Copilot."""

import os
import uuid
from datetime import date


def generate_call_id() -> str:
    """Generate a unique call ID (first 8 chars of a UUID4 hex string)."""
    return uuid.uuid4().hex[:8]


def extract_title_from_filename(filename: str) -> str:
    """Derive a human-readable title from a transcript filename.

    Strips the file extension, replaces underscores and hyphens with spaces,
    and title-cases the result.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    return name.replace("_", " ").replace("-", " ").title()


def get_current_date_iso() -> str:
    """Return the current date as an ISO format string (YYYY-MM-DD)."""
    return date.today().isoformat()
