"""Retrieval component for RAG pipeline."""

import re
import math
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Any

import faiss

from .utils import get_logger

logger = get_logger(__name__)


class FAISSRetriever:
    """FAISS-based retriever for document chunks."""

    def __init__(
        self,
        embeddings: np.ndarray,
        chunks: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        dimension: int = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        normalize_embeddings_for_faiss: bool = False
    ):
        """
        Initialize FAISS retriever.

        Args:
            embeddings: Matrix of chunk embeddings
            chunks: List of chunk texts
            metadata: Chunk metadata
            dimension: Embedding dimension
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            normalize_embeddings_for_faiss: Use cosine similarity
        """

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings and chunks must have same length: "
                f"{len(embeddings)} vs {len(chunks)}"
            )

        self.chunks = chunks
        self.metadata = metadata or [{} for _ in chunks]

        if len(self.metadata) != len(self.chunks):
            raise ValueError("metadata and chunks must have same length")

        if dimension is None:
            dimension = embeddings.shape[1]

        self.dimension = dimension
        self.normalize_embeddings_for_faiss = normalize_embeddings_for_faiss

        self._stored_embeddings = embeddings.astype("float32")

        if self.normalize_embeddings_for_faiss:
            self._stored_embeddings = self._normalize_vectors(
                self._stored_embeddings
            )

        logger.info(
            f"Building FAISS index with {len(chunks)} chunks, "
            f"dimension={dimension}"
        )

        if self.normalize_embeddings_for_faiss:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(self._stored_embeddings)

        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        self._build_bm25_index()
        self._build_parent_lookup()

        logger.info(f"FAISS index built. Index size: {self.index.ntotal}")

    def _build_parent_lookup(self):
        """
        Build mapping from parent chunk id -> child indices.

        Assumes metadata may contain:
            - parent_id
            - chunk_level ('parent' or 'child')
        """

        self.parent_to_children = defaultdict(list)
        self.child_to_parent = {}

        for idx, meta in enumerate(self.metadata):
            parent_id = meta.get("parent_id")

            if parent_id is not None:
                self.parent_to_children[parent_id].append(idx)
                self.child_to_parent[idx] = parent_id

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        vectors = vectors.astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        query_text: Optional[str] = None,
        retriever_type: str = "similarity",
        hybrid_alpha: float = 0.5,
        hybrid_candidate_k: int = 50,
        score_threshold: Optional[float] = None,
        min_retrieved: int = 1,
        max_retrieved: Optional[int] = None,
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Main retrieval entrypoint.
        """

        if retriever_type == "mmr":
            return self.retrieve_mmr(query_embedding, top_k)

        if retriever_type == "hybrid":
            if query_text is None:
                raise ValueError(
                    "query_text is required when retriever_type='hybrid'"
                )

            return self.retrieve_hybrid(
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                alpha=hybrid_alpha,
                candidate_k=hybrid_candidate_k,
                score_threshold=score_threshold,
                min_retrieved=min_retrieved,
                max_retrieved=max_retrieved,
            )

        query_embedding = np.array([query_embedding], dtype="float32")

        if self.normalize_embeddings_for_faiss:
            query_embedding = self._normalize_vectors(query_embedding)

        retrieve_k = (
            max_retrieved
            if score_threshold is not None and max_retrieved is not None
            else top_k
        )

        retrieve_k = min(retrieve_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, retrieve_k)

        retrieved_indices = indices[0].tolist()
        retrieved_scores = scores[0].tolist()
        retrieved_chunks = [self.chunks[idx] for idx in retrieved_indices]

        return retrieved_chunks, retrieved_scores, retrieved_indices

    def expand_child_results_to_parents(
        self,
        chunk_indices: List[int],
        scores: List[float],
        max_parents: int = 5,
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Convert retrieved child chunks into parent chunks.

        Keeps highest scoring child per parent.
        """

        parent_best_scores = {}
        parent_best_indices = {}

        for idx, score in zip(chunk_indices, scores):
            meta = self.metadata[idx]

            parent_id = meta.get("parent_id")
            parent_index = meta.get("parent_index")

            if parent_id is None or parent_index is None:
                parent_id = idx
                parent_index = idx

            if (
                parent_id not in parent_best_scores
                or score > parent_best_scores[parent_id]
            ):
                parent_best_scores[parent_id] = score
                parent_best_indices[parent_id] = parent_index

        ranked = sorted(
            parent_best_indices.items(),
            key=lambda x: parent_best_scores[x[0]],
            reverse=True,
        )

        ranked = ranked[:max_parents]

        final_indices = [parent_idx for _, parent_idx in ranked]
        final_scores = [parent_best_scores[parent_id] for parent_id, _ in ranked]
        final_chunks = [self.chunks[idx] for idx in final_indices]

        return final_chunks, final_scores, final_indices

    def retrieve_mmr(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        diversity_weight: float = 0.5,
    ) -> Tuple[List[str], List[float], List[int]]:

        query_embedding = np.array([query_embedding], dtype="float32")

        if self.normalize_embeddings_for_faiss:
            query_embedding = self._normalize_vectors(query_embedding)

        retrieve_k = min(top_k * 10, self.index.ntotal)

        distances, indices = self.index.search(query_embedding, retrieve_k)

        candidate_indices = indices[0].tolist()
        candidate_scores = distances[0].tolist()

        if self.normalize_embeddings_for_faiss:
            relevance_scores = np.array(candidate_scores)
        else:
            relevance_scores = np.array(
                [1.0 / (1.0 + d) for d in candidate_scores]
            )

        candidate_embeddings = np.array(
            [self._stored_embeddings[idx] for idx in candidate_indices],
            dtype="float32",
        )

        selected_positions = []
        remaining_positions = list(range(len(candidate_indices)))

        best_position = int(np.argmax(relevance_scores))
        selected_positions.append(best_position)
        remaining_positions.remove(best_position)

        while len(selected_positions) < top_k and remaining_positions:
            best_score = None
            best_remaining_position = None

            for candidate_position in remaining_positions:
                relevance = relevance_scores[candidate_position]
                candidate_embedding = candidate_embeddings[candidate_position]

                diversity_penalty = 0.0

                for selected_position in selected_positions:
                    selected_embedding = candidate_embeddings[selected_position]

                    similarity = np.dot(
                        candidate_embedding,
                        selected_embedding,
                    ) / (
                        np.linalg.norm(candidate_embedding)
                        * np.linalg.norm(selected_embedding)
                        + 1e-8
                    )

                    diversity_penalty = max(diversity_penalty, similarity)

                mmr_score = relevance - diversity_weight * diversity_penalty

                if best_score is None or mmr_score > best_score:
                    best_score = mmr_score
                    best_remaining_position = candidate_position

            if best_remaining_position is None:
                break

            selected_positions.append(best_remaining_position)
            remaining_positions.remove(best_remaining_position)

        selected_indices = [candidate_indices[pos] for pos in selected_positions]

        retrieved_chunks = [self.chunks[idx] for idx in selected_indices]
        retrieved_scores = [
            float(relevance_scores[pos])
            for pos in selected_positions
        ]

        return retrieved_chunks, retrieved_scores, selected_indices

    def _tokenize(self, text: str) -> List[str]:
        """
        Better tokenizer for technical documentation.

        Handles:
        - RFGridSearch
        - run_evals
        - snake_case
        - camelCase
        - class/function names
        """

        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = text.replace("_", " ")
        text = text.replace("-", " ")

        tokens = re.findall(r"\b\w+\b", text.lower())

        stopwords = {
            "the", "and", "for", "with", "that", "this",
            "from", "are", "was", "were", "been", "have",
            "has", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "is",
            "as", "in", "of", "to", "by", "on", "at",
            "be", "or", "an", "but", "not", "if", "it",
            "what", "how", "when", "which", "where", "why",
            "who"
        }

        return [
            token
            for token in tokens
            if len(token) > 1 and token not in stopwords
        ]

    def _build_bm25_index(self):
        logger.info(f"Building BM25 index with {len(self.chunks)} chunks")

        self._bm25_doc_tokens = [
            self._tokenize(chunk)
            for chunk in self.chunks
        ]

        self._bm25_doc_lengths = [
            len(tokens)
            for tokens in self._bm25_doc_tokens
        ]

        self._bm25_avgdl = (
            sum(self._bm25_doc_lengths) / len(self._bm25_doc_lengths)
            if self._bm25_doc_lengths
            else 0.0
        )

        self._bm25_doc_freqs: Dict[str, int] = {}
        self._bm25_term_freqs = []

        for tokens in self._bm25_doc_tokens:
            term_freq = Counter(tokens)
            self._bm25_term_freqs.append(term_freq)

            for term in term_freq.keys():
                self._bm25_doc_freqs[term] = (
                    self._bm25_doc_freqs.get(term, 0) + 1
                )

        logger.info(
            f"BM25 index built. Vocabulary size: "
            f"{len(self._bm25_doc_freqs)}"
        )

    def _bm25_scores(self, query_text: str) -> np.ndarray:
        query_terms = self._tokenize(query_text)

        scores = np.zeros(len(self.chunks), dtype="float32")

        if not query_terms or not self.chunks:
            return scores

        num_docs = len(self.chunks)

        for term in query_terms:
            doc_freq = self._bm25_doc_freqs.get(term, 0)

            if doc_freq == 0:
                continue

            idf = math.log(
                1 + (num_docs - doc_freq + 0.5)
                / (doc_freq + 0.5)
            )

            for doc_idx, term_freq in enumerate(self._bm25_term_freqs):
                freq = term_freq.get(term, 0)

                if freq == 0:
                    continue

                doc_len = self._bm25_doc_lengths[doc_idx]

                denom = freq + self.bm25_k1 * (
                    1
                    - self.bm25_b
                    + self.bm25_b * doc_len
                    / (self._bm25_avgdl + 1e-8)
                )

                scores[doc_idx] += (
                    idf * (freq * (self.bm25_k1 + 1))
                    / (denom + 1e-8)
                )

        return scores

    def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores

        min_score = float(np.min(scores))
        max_score = float(np.max(scores))

        if max_score - min_score < 1e-8:
            return np.zeros_like(scores, dtype="float32")

        return (
            (scores - min_score)
            / (max_score - min_score)
        ).astype("float32")

    def retrieve_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.5,
        candidate_k: int = 50,
        score_threshold: Optional[float] = None,
        min_retrieved: int = 1,
        max_retrieved: Optional[int] = None,
    ) -> Tuple[List[str], List[float], List[int]]:

        query_embedding = np.array([query_embedding], dtype="float32")

        if self.normalize_embeddings_for_faiss:
            query_embedding = self._normalize_vectors(query_embedding)

        candidate_k = min(candidate_k, self.index.ntotal)

        dense_scores, dense_indices = self.index.search(
            query_embedding,
            candidate_k,
        )

        dense_indices = dense_indices[0].tolist()
        dense_scores = dense_scores[0].tolist()

        if self.normalize_embeddings_for_faiss:
            dense_scores_by_idx = {
                idx: score
                for idx, score in zip(dense_indices, dense_scores)
            }
        else:
            dense_scores_by_idx = {
                idx: 1.0 / (1.0 + distance)
                for idx, distance in zip(dense_indices, dense_scores)
            }

        bm25_scores = self._bm25_scores(query_text)

        bm25_candidate_indices = np.argsort(bm25_scores)[::-1][
            :candidate_k
        ].tolist()

        candidate_indices = list(
            dict.fromkeys(dense_indices + bm25_candidate_indices)
        )

        dense_candidate_scores = np.array(
            [dense_scores_by_idx.get(idx, 0.0) for idx in candidate_indices],
            dtype="float32",
        )

        bm25_candidate_scores = np.array(
            [bm25_scores[idx] for idx in candidate_indices],
            dtype="float32",
        )

        dense_candidate_scores = self._min_max_normalize(
            dense_candidate_scores
        )

        bm25_candidate_scores = self._min_max_normalize(
            bm25_candidate_scores
        )

        combined_scores = (
            alpha * dense_candidate_scores
            + (1 - alpha) * bm25_candidate_scores
        )

        ranked_positions = np.argsort(combined_scores)[::-1]

        if score_threshold is not None:
            selected_positions = [
                pos
                for pos in ranked_positions
                if combined_scores[pos] >= score_threshold
            ]

            if len(selected_positions) < min_retrieved:
                selected_positions = ranked_positions[:min_retrieved].tolist()

            if max_retrieved is not None:
                selected_positions = selected_positions[:max_retrieved]
        else:
            selected_positions = ranked_positions[:top_k].tolist()

        selected_indices = [
            candidate_indices[pos]
            for pos in selected_positions
        ]

        selected_scores = [
            float(combined_scores[pos])
            for pos in selected_positions
        ]

        retrieved_chunks = [
            self.chunks[idx]
            for idx in selected_indices
        ]

        return retrieved_chunks, selected_scores, selected_indices