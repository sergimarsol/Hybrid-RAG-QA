"""Configuration management for RAG pipeline."""

import json
import os
from pathlib import Path
from typing import Dict, Any


class RAGConfig:
    """RAG Pipeline Configuration."""

    def __init__(self, **kwargs):
        """Initialize configuration with defaults or provided values."""

        # ============================================================
        # Document processing
        # ============================================================

        # Child chunk settings.
        # Child chunks are used for retrieval and reranking.
        self.chunk_size = kwargs.get("chunk_size", 512)
        self.chunk_overlap = kwargs.get("chunk_overlap", 200)

        # Parent-child retrieval.
        # Child chunks are indexed.
        # Parent chunks are returned after reranking.
        self.use_parent_child_chunking = kwargs.get(
            "use_parent_child_chunking",
            False,
        )

        # Maximum approximate token size for parent chunks.
        self.parent_max_tokens = kwargs.get(
            "parent_max_tokens",
            1000,
        )

        # ============================================================
        # Embedding
        # ============================================================

        self.embedding_model = kwargs.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        # ============================================================
        # Retrieval
        # ============================================================

        # Supported:
        # - similarity: FAISS dense retrieval only
        # - mmr: dense retrieval with diversity
        # - hybrid: BM25 + dense fusion
        self.retriever_type = kwargs.get(
            "retriever_type",
            "hybrid",
        )

        # Final number of returned chunks/sources.
        self.top_k = kwargs.get("top_k", 2)

        # Retrieve-wide pool before reranking.
        self.retrieval_pool_k = kwargs.get(
            "retrieval_pool_k",
            30,
        )

        # Multi-query retrieval.
        # When enabled, the pipeline asks the LLM for complementary queries
        # and merges the resulting candidates before reranking.
        self.use_multi_query_retrieval = kwargs.get(
            "use_multi_query_retrieval",
            False,
        )

        self.multi_query_count = kwargs.get(
            "multi_query_count",
            1,
        )

        # ============================================================
        # Hybrid retrieval parameters
        # ============================================================

        # hybrid_alpha controls dense-vs-BM25 weighting:
        #   0.0 = BM25 only inside hybrid scoring
        #   1.0 = dense only inside hybrid scoring
        #   0.5 = equal blend
        self.hybrid_alpha = kwargs.get("hybrid_alpha", 0.15)

        # Number of dense/BM25 candidates merged before scoring.
        self.hybrid_candidate_k = kwargs.get(
            "hybrid_candidate_k",
            50,
        )

        # ============================================================
        # BM25 parameters used only by hybrid retrieval
        # ============================================================

        self.bm25_k1 = kwargs.get("bm25_k1", 1.5)
        self.bm25_b = kwargs.get("bm25_b", 0.75)

        # ============================================================
        # FAISS embedding normalization
        # ============================================================

        self.normalize_embeddings_for_faiss = kwargs.get(
            "normalize_embeddings_for_faiss",
            False,
        )

        # ============================================================
        # Threshold-based retrieval
        # ============================================================

        self.use_retrieval_threshold = kwargs.get(
            "use_retrieval_threshold",
            False,
        )

        self.retrieval_score_threshold = kwargs.get(
            "retrieval_score_threshold",
            0.5,
        )

        self.min_retrieved = kwargs.get("min_retrieved", 1)

        self.max_retrieved = kwargs.get(
            "max_retrieved",
            self.top_k,
        )

        # ============================================================
        # Reranking
        # ============================================================

        # Cross-encoder reranking.
        self.use_reranker = kwargs.get("use_reranker", False)

        self.reranker_model = kwargs.get(
            "reranker_model",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        # ============================================================
        # LLM
        # ============================================================

        self.llm_model = kwargs.get(
            "llm_model",
            "api-gpt-oss-120b",
        )

        self.llm_temperature = kwargs.get(
            "llm_temperature",
            0.0,
        )

        self.llm_max_tokens = kwargs.get(
            "llm_max_tokens",
            500,
        )

        self.llm_api_base = kwargs.get(
            "llm_api_base",
            os.environ.get("LLM_API_BASE", "https://api.openai.com/v1"),
        )

        self.llm_api_key = kwargs.get("llm_api_key", None)

        # ============================================================
        # Context budget
        # ============================================================

        self.max_context_tokens = kwargs.get(
            "max_context_tokens",
            2000,
        )

        # ============================================================
        # Vector store
        # ============================================================

        self.vector_store_type = kwargs.get(
            "vector_store_type",
            "faiss",
        )

        # ============================================================
        # Corpus path
        # ============================================================

        corpus_path_arg = kwargs.get("corpus_path", None)

        if corpus_path_arg is None:
            project_root = Path(__file__).parent.parent
            corpus_path_arg = (
                project_root / "data" / "sourcedocs"
            ).resolve()
        else:
            corpus_path_arg = Path(corpus_path_arg).resolve()

        self.corpus_path = str(corpus_path_arg)

        # ============================================================
        # Prompt template
        # ============================================================

        self.system_prompt = kwargs.get(
            "system_prompt",
            self._default_system_prompt(),
        )

    @staticmethod
    def _default_system_prompt() -> str:
        return """You are a helpful assistant that answers questions about RapidFire AI documentation.
Use only the provided context to answer questions. If the answer is not in the context, say so clearly.
Be concise and accurate. Cite the relevant sections when possible."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""

        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """Create config from dictionary."""

        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "RAGConfig":
        """Load config from JSON file."""

        with open(json_path, "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def save_json(self, json_path: str) -> None:
        """Save config to JSON file."""

        Path(json_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_default_config() -> RAGConfig:
    """Get default RAG configuration."""

    return RAGConfig()