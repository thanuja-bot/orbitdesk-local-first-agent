from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AgentConfig
from .data import load_corpus
from .types import Passage


STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "for", "from", "how", "i", "in",
    "is", "it", "my", "of", "on", "or", "our", "the", "their", "to", "we",
    "what", "with", "you", "after", "still", "this", "that",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    ]


@dataclass
class RetrievalResult:
    passages: list[Passage]
    latency_ms: float
    backend: str


class LocalRetriever:
    """Local semantic retriever with an explicit offline lexical index.

    If the pinned embedding model is present, it is loaded through
    Hugging Face Transformers. The lexical index is deterministic and is used
    by tests and by the application before the optional model download.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.corpus = load_corpus(config.data_dir)
        self.backend = "lexical"
        self._model = None
        self._tokenizer = None
        self._doc_vectors: np.ndarray | None = None
        self._idf: dict[str, float] = {}
        self._build_lexical_index()
        if config.use_huggingface:
            self._try_load_hf()

    def _build_lexical_index(self) -> None:
        docs = [_tokens(p.text) for p in self.corpus]
        document_frequency = Counter(token for doc in docs for token in set(doc))
        count = len(docs) or 1
        self._idf = {
            token: math.log((1 + count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }

    def _try_load_hf(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            model_path = self.config.model_dir / "embeddings"
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
            ).to(self.config.model.device)
            self._model.eval()
            self.backend = "huggingface"
            self._doc_vectors = np.vstack(
                [self._embed(p.text, torch) for p in self.corpus]
            )
        except (ImportError, OSError, RuntimeError, ValueError):
            # No network call is attempted. A missing optional model is an
            # explicit, deterministic lexical mode rather than a fake remote
            # fallback.
            self._model = None
            self._tokenizer = None

    def _embed(self, text: str, torch_module) -> np.ndarray:
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=384,
        )
        encoded = {key: value.to(self.config.model.device) for key, value in encoded.items()}
        with torch_module.no_grad():
            output = self._model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
        pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vector = pooled[0].cpu().numpy()
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def _lexical_score(self, query: str, passage: Passage) -> float:
        query_terms = Counter(_tokens(query))
        passage_terms = Counter(_tokens(passage.text))
        score = 0.0
        for term, frequency in query_terms.items():
            if term in passage_terms:
                score += (1 + math.log(frequency)) * self._idf.get(term, 1.0)
        normalized = score / max(math.sqrt(sum(v * v for v in query_terms.values())), 1.0)
        # Current docs win ties and cases add useful multi-document context.
        return normalized + (passage.priority / 10000.0)

    def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        started = time.perf_counter()
        limit = top_k or self.config.top_k
        if self.backend == "huggingface" and self._model is not None and self._doc_vectors is not None:
            import torch

            vector = self._embed(query, torch)
            scores = self._doc_vectors @ vector
            ranked = sorted(
                zip(scores.tolist(), self.corpus),
                key=lambda item: (item[0], item[1].priority),
                reverse=True,
            )
        else:
            ranked = sorted(
                ((self._lexical_score(query, passage), passage) for passage in self.corpus),
                key=lambda item: (item[0], item[1].priority),
                reverse=True,
            )
        results = [
            passage.model_copy(update={"score": round(float(score), 4)})
            for score, passage in ranked[:limit]
            if score > 0
        ]
        return RetrievalResult(
            passages=results,
            latency_ms=(time.perf_counter() - started) * 1000,
            backend=self.backend,
        )