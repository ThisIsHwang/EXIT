"""Base interface for document compressors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SearchResult:
    """Retrieved source document or document fragment."""

    evi_id: int
    docid: int
    title: str
    text: str
    score: Optional[float] = None


class BaseCompressor(ABC):
    """Abstract base class for document compressors."""

    @abstractmethod
    def compress(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Compress documents based on query relevance.

        Args:
            query: Input question
            documents: List of documents to compress

        Returns:
            Compressed documents with relevance scores
        """
        raise NotImplementedError
