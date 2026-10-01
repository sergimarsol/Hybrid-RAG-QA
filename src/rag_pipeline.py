"""Main RAG pipeline orchestrator."""

from typing import List, Dict, Optional
import json
from pathlib import Path
from datetime import datetime
from sentence_transformers import CrossEncoder

from .config import RAGConfig, get_default_config
from .data_loader import CorpusLoader
from .embedding import EmbeddingModel
from .retriever import FAISSRetriever
from .generator import ContextAssembler, SimpleGenerator
from .utils import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or get_default_config()

        self.loader = None
        self.embedding_model = None
        self.retriever = None
        self.context_assembler = None
        self.generator = None
        self.reranker = None
        self.query_expander = None

        self.documents = []
        self.chunks = []
        self.chunk_metadata = []

    def build(self) -> None:
        logger.info(f"Initializing RAG pipeline with config: {self.config.to_dict()}")

        self.loader = CorpusLoader(
            corpus_path=self.config.corpus_path,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            use_parent_child_chunking=self.config.use_parent_child_chunking,
            parent_max_tokens=self.config.parent_max_tokens,
        )

        self.documents = self.loader.load_and_chunk()
        self.chunks = [doc.page_content for doc in self.documents]
        self.chunk_metadata = [doc.metadata for doc in self.documents]

        logger.info(f"Loaded {len(self.chunks)} chunks")
        logger.info(f"Chunk stats: {self.loader.get_chunk_stats()}")

        self.embedding_model = EmbeddingModel(self.config.embedding_model)
        embeddings = self.embedding_model.embed_chunk_list(self.chunks)

        self.retriever = FAISSRetriever(
            embeddings=embeddings,
            chunks=self.chunks,
            metadata=self.chunk_metadata,
            dimension=embeddings.shape[1],
            bm25_k1=self.config.bm25_k1,
            bm25_b=self.config.bm25_b,
            normalize_embeddings_for_faiss=self.config.normalize_embeddings_for_faiss,
        )

        if self.config.use_reranker:
            logger.info(f"Loading reranker: {self.config.reranker_model}")
            self.reranker = CrossEncoder(self.config.reranker_model)

        self.context_assembler = ContextAssembler(
            system_prompt=self.config.system_prompt,
            max_context_tokens=self.config.max_context_tokens,
        )

        self.generator = SimpleGenerator(
            model_name=self.config.llm_model,
            api_base=self.config.llm_api_base,
            api_key=self.config.llm_api_key,
        )

        logger.info("Pipeline built successfully")

    def _dedupe_queries(self, queries: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for query in queries:
            normalized = query.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(query.strip())
        return deduped

    def _expand_queries(self, question: str) -> List[str]:
        if not self.config.use_multi_query_retrieval:
            return [question]

        variant_count = max(0, int(self.config.multi_query_count))
        if variant_count <= 0:
            return [question]

        variants = self.generator.generate_query_variants(
            question=question,
            num_queries=variant_count,
        )
        return self._dedupe_queries([question, *variants])

    def _merge_multi_query_results(
        self,
        retrieved_sets: List[tuple[List[str], List[float], List[int]]],
    ) -> tuple[List[str], List[float], List[int]]:
        if len(retrieved_sets) == 1:
            return retrieved_sets[0]

        best_by_index = {}

        for chunks, scores, indices in retrieved_sets:
            for chunk, score, idx in zip(chunks, scores, indices):
                normalized_score = float(score)

                if self.config.retriever_type == "similarity" and not self.config.normalize_embeddings_for_faiss:
                    normalized_score = 1.0 / (1.0 + float(score))
                elif self.config.retriever_type == "similarity" and self.config.normalize_embeddings_for_faiss:
                    normalized_score = float(score)

                current = best_by_index.get(idx)
                if current is None or normalized_score > current["score"]:
                    best_by_index[idx] = {
                        "chunk": chunk,
                        "score": normalized_score,
                        "index": idx,
                    }

        ranked = sorted(
            best_by_index.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        if self.config.use_reranker:
            limit = self.config.retrieval_pool_k
        else:
            limit = self.config.top_k

        ranked = ranked[:limit]

        return (
            [item["chunk"] for item in ranked],
            [float(item["score"]) for item in ranked],
            [int(item["index"]) for item in ranked],
        )

    def _retrieve_candidates(
        self,
        question: str,
        query_embedding,
        first_stage_k: int,
    ) -> tuple[List[str], List[float], List[int]]:
        return self.retriever.retrieve(
            query_embedding=query_embedding,
            top_k=first_stage_k,
            query_text=question if self.config.retriever_type == "hybrid" else None,
            retriever_type=self.config.retriever_type,
            hybrid_alpha=self.config.hybrid_alpha,
            hybrid_candidate_k=self.config.hybrid_candidate_k,
            score_threshold=(
                self.config.retrieval_score_threshold
                if self.config.use_retrieval_threshold
                else None
            ),
            min_retrieved=self.config.min_retrieved,
            max_retrieved=self.config.max_retrieved,
        )
    def _filter_child_results(
        self,
        chunks: List[str],
        scores: List[float],
        indices: List[int],
    ):
        """Keep child chunks when parent-child chunking is enabled."""
        if not self.config.use_parent_child_chunking:
            return chunks, scores, indices

        filtered = [
            (chunk, score, idx)
            for chunk, score, idx in zip(chunks, scores, indices)
            if self.chunk_metadata[idx].get("chunk_level") == "child"
        ]

        if not filtered:
            logger.warning("No child chunks found after retrieval; falling back to raw retrieved chunks")
            return chunks, scores, indices

        return (
            [x[0] for x in filtered],
            [x[1] for x in filtered],
            [x[2] for x in filtered],
        )

    def rerank_chunks(
        self,
        question: str,
        retrieved_chunks: List[str],
        retrieved_scores: List[float],
        chunk_indices: List[int],
    ):
        if not self.config.use_reranker or self.reranker is None:
            return retrieved_chunks[:self.config.top_k], retrieved_scores[:self.config.top_k], chunk_indices[:self.config.top_k]

        logger.info(f"Cross-encoder reranking {len(retrieved_chunks)} child candidates")

        pairs = [(question, chunk) for chunk in retrieved_chunks]
        cross_scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(retrieved_chunks, cross_scores, chunk_indices),
            key=lambda x: x[1],
            reverse=True,
        )[:self.config.top_k]

        return (
            [x[0] for x in ranked],
            [float(x[1]) for x in ranked],
            [x[2] for x in ranked],
        )

    def _get_parent_context_for_child_sources(
        self,
        child_indices: List[int],
        child_scores: List[float],
    ):
        """
        Expand selected child sources to parent chunks for generation context only.
        Does NOT change the source spans used for retrieval evaluation.
        """
        if not self.config.use_parent_child_chunking:
            return [self.chunks[i] for i in child_indices], child_indices

        parent_seen = set()
        parent_indices = []

        for child_idx in child_indices:
            meta = self.chunk_metadata[child_idx]
            parent_index = meta.get("parent_index")

            if parent_index is None:
                parent_index = child_idx

            if parent_index not in parent_seen:
                parent_seen.add(parent_index)
                parent_indices.append(parent_index)

        parent_chunks = [self.chunks[i] for i in parent_indices]

        return parent_chunks, parent_indices

    def _log_retrieved_chunks(
        self,
        question: str,
        queries: List[str],
        child_chunks: List[str],
        child_scores: List[float],
        child_indices: List[int],
        parent_context_chunks: List[str],
        parent_context_indices: List[int],
    ) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "retrieved_chunks.jsonl"

        child_entries = []
        for chunk, score, idx in zip(child_chunks, child_scores, child_indices):
            meta = self.chunk_metadata[idx]
            child_entries.append({
                "index": int(idx),
                "score": float(score),
                "filename": meta.get("filename", "unknown"),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "section_title": meta.get("section_title", ""),
                "chunk_level": meta.get("chunk_level", ""),
                "text": chunk,
            })

        parent_entries = []
        for chunk, idx in zip(parent_context_chunks, parent_context_indices):
            meta = self.chunk_metadata[idx]
            parent_entries.append({
                "index": int(idx),
                "filename": meta.get("filename", "unknown"),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "section_title": meta.get("section_title", ""),
                "chunk_level": meta.get("chunk_level", ""),
                "text": chunk,
            })

        record = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "queries": queries,
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "top_k": self.config.top_k,
                "retriever_type": self.config.retriever_type,
                "hybrid_alpha": self.config.hybrid_alpha,
                "hybrid_candidate_k": self.config.hybrid_candidate_k,
                "retrieval_pool_k": self.config.retrieval_pool_k,
                "use_reranker": self.config.use_reranker,
                "use_parent_child_chunking": self.config.use_parent_child_chunking,
                "use_multi_query_retrieval": self.config.use_multi_query_retrieval,
                "multi_query_count": self.config.multi_query_count,
            },
            "child_sources_for_eval": child_entries,
            "parent_context_for_generation": parent_entries,
        }

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def query(self, question: str) -> Dict:
        if self.retriever is None:
            raise RuntimeError("Pipeline not built. Call build() first.")

        logger.info(f"Processing query: {question}")

        query_embedding = self.embedding_model.embed_query(question)

        first_stage_k = (
            self.config.retrieval_pool_k
            if self.config.use_reranker
            else self.config.top_k
        )

        queries = self._expand_queries(question)
        logger.info(f"Retrieval queries: {queries}")

        retrieved_sets = []
        for query_text in queries:
            query_embedding = self.embedding_model.embed_query(query_text)
            retrieved_sets.append(
                self._retrieve_candidates(
                    question=query_text,
                    query_embedding=query_embedding,
                    first_stage_k=first_stage_k,
                )
            )

        retrieved_chunks, retrieved_scores, chunk_indices = self._merge_multi_query_results(retrieved_sets)

        retrieved_chunks, retrieved_scores, chunk_indices = self._filter_child_results(
            retrieved_chunks,
            retrieved_scores,
            chunk_indices,
        )

        child_chunks, child_scores, child_indices = self.rerank_chunks(
            question=question,
            retrieved_chunks=retrieved_chunks,
            retrieved_scores=retrieved_scores,
            chunk_indices=chunk_indices,
        )

        parent_context_chunks, parent_context_indices = self._get_parent_context_for_child_sources(
            child_indices,
            child_scores,
        )

        self._log_retrieved_chunks(
            question=question,
            queries=queries,
            child_chunks=child_chunks,
            child_scores=child_scores,
            child_indices=child_indices,
            parent_context_chunks=parent_context_chunks,
            parent_context_indices=parent_context_indices,
        )

        parent_metadata = [
            self.chunk_metadata[idx]
            for idx in parent_context_indices
        ]

        context, token_usage = self.context_assembler.assemble(
            question=question,
            retrieved_chunks=parent_context_chunks,
            chunk_metadata=parent_metadata,
        )

        answer = self.generator.generate(
            context,
            max_tokens=self.config.llm_max_tokens,
        )

        # IMPORTANT:
        # Use child chunks as sources for retrieval evaluation.
        sources = []
        for idx in child_indices:
            meta = self.chunk_metadata[idx]
            sources.append({
                "filename": meta.get("filename", "unknown"),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
            })

        # Use parent chunks for generation faithfulness context.
        retrieved_context_parts = []
        for chunk, idx in zip(parent_context_chunks, parent_context_indices):
            meta = self.chunk_metadata[idx]
            filename = meta.get("filename", "unknown")
            section_title = meta.get("section_title", "")
            section_path = meta.get("section_path", section_title)

            retrieved_context_parts.append(
                f"[Source: {filename}]\n"
                f"[Section: {section_path}]\n"
                f"{chunk}"
            )

        retrieved_context = "\n\n".join(retrieved_context_parts)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_context": retrieved_context,
            "token_usage": token_usage,
            "retrieved_chunks_count": len(child_chunks),
            "scores": child_scores,
        }

    def batch_query(self, questions: List[str]) -> List[Dict]:
        return [self.query(q) for q in questions]


def create_pipeline(config: Optional[RAGConfig] = None) -> RAGPipeline:
    return RAGPipeline(config)