from dataclasses import dataclass,field
from typing import Any

@dataclass
class KnowledgeDocument:
    source: str
    content: str
    metadata: dict[str,Any] = field(default_factory=dict)

@dataclass
class KnowledgeChunk:
    content: str
    source: str
    heading: str
    metadata: dict[str,Any] = field(default_factory=dict)
    score: float | None = None
    authority_score: float | None = None

