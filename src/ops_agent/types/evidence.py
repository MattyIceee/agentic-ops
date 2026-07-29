"""Evidence item model for LLM-gathered information."""

from pydantic import BaseModel


class EvidenceItem(BaseModel):
    """A piece of evidence gathered from an external source."""

    source: str
    url: str | None
    text: str
