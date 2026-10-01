#!/usr/bin/env python3
"""
Main CLI entry point for RAG pipeline.

Required autograder usage:
    python main.py \
        --input <path> \
        --output <path> \
        --corpus-dir <path> \
        --apikey-txt <path_to_txt> \
        --generation-model <name>
"""

import argparse
import os
import json
import sys
from pathlib import Path

from src.rag_pipeline import create_pipeline
from src.config import RAGConfig
from src.utils import get_logger

logger = get_logger(__name__)


def load_input_file(input_file: str) -> list:
    """
    Load input JSON file with questions.

    Expected format:
    [
        {"question_id": 1, "question": "..."},
        {"question_id": 2, "question": "..."},
        ...
    ]
    """

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON list")

    logger.info(f"Loaded {len(data)} questions from {input_file}")

    return data


def load_api_key(apikey_txt: str | None) -> str | None:
    """Load TritonAI API key from a text file."""

    if apikey_txt is None:
        return None

    key_path = Path(apikey_txt).expanduser()

    if not key_path.exists():
        raise FileNotFoundError(f"API key file not found: {key_path}")

    return key_path.read_text(encoding="utf-8").strip()


def save_output_file(results: list, output_file: str) -> None:
    """Save results to output JSON file."""

    output_data = []

    for result in results:
        output_item = {
            "question_id": result.get("question_id"),
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "retrieved_context": result.get("retrieved_context", ""),
        }

        output_data.append(output_item)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved {len(output_data)} results to {output_file}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="RAG Pipeline for RapidFire AI Documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ============================================================
    # Required autograder arguments
    # ============================================================

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file with questions",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for answers",
    )

    parser.add_argument(
        "--corpus-dir",
        required=True,
        help="Path to corpus directory containing RST files",
    )

    parser.add_argument(
        "--apikey-txt",
        required=True,
        help="Path to text file containing TritonAI API key",
    )

    parser.add_argument(
        "--generation-model",
        required=True,
        help="Gateway model id for answer generation",
    )

    # ============================================================
    # Optional config
    # ============================================================

    parser.add_argument(
        "--config",
        default=None,
        help="Optional config JSON file. CLI defaults are ignored if this is used.",
    )

    # ============================================================
    # Chunking
    # ============================================================

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Child chunk size in approximate tokens. Default: 512",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Child chunk overlap in approximate tokens. Default: 200",
    )

    parser.add_argument(
        "--use-parent-child-chunking",
        action="store_true",
        help=(
            "Enable parent-child chunking. "
            "Child chunks are indexed and parent sections are returned."
        ),
    )

    parser.add_argument(
        "--parent-max-tokens",
        type=int,
        default=1000,
        help="Maximum parent section size in tokens. Default: 1000",
    )

    # ============================================================
    # Embedding
    # ============================================================

    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name",
    )

    parser.add_argument(
        "--normalize-embeddings-for-faiss",
        action="store_true",
        help="Normalize embeddings before FAISS search",
    )

    # ============================================================
    # Retrieval
    # ============================================================

    parser.add_argument(
        "--retriever-type",
        choices=["similarity", "mmr", "hybrid"],
        default="hybrid",
        help="Retriever strategy. Default: hybrid",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Final number of retrieved chunks. Default: 2",
    )

    parser.add_argument(
        "--retrieval-pool-k",
        type=int,
        default=30,
        help="First-stage retrieval pool size. Default: 30",
    )

    parser.add_argument(
        "--use-multi-query-retrieval",
        action="store_true",
        help="Enable LLM-generated multi-query retrieval",
    )

    parser.add_argument(
        "--multi-query-count",
        type=int,
        default=1,
        help="Number of generated retrieval queries. Default: 1",
    )

    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.15,
        help="Hybrid dense retrieval weight. Default: 0.15",
    )

    parser.add_argument(
        "--hybrid-candidate-k",
        type=int,
        default=50,
        help="Merged dense/BM25 candidate pool size. Default: 50",
    )

    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter. Default: 1.5",
    )

    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.75,
        help="BM25 b parameter. Default: 0.75",
    )

    # ============================================================
    # Threshold retrieval
    # ============================================================

    parser.add_argument(
        "--use-retrieval-threshold",
        action="store_true",
        help="Enable threshold-based retrieval",
    )

    parser.add_argument(
        "--retrieval-score-threshold",
        type=float,
        default=0.5,
        help="Retrieval threshold score. Default: 0.5",
    )

    parser.add_argument(
        "--min-retrieved",
        type=int,
        default=1,
        help="Minimum retrieved chunks. Default: 1",
    )

    parser.add_argument(
        "--max-retrieved",
        type=int,
        default=2,
        help="Maximum retrieved chunks. Default: 2",
    )

    # ============================================================
    # Reranking
    # ============================================================

    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable cross-encoder reranking",
    )

    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder reranker model",
    )

    # ============================================================
    # Optional LLM/API overrides
    # ============================================================

    parser.add_argument(
        "--llm-api-base",
        default=os.environ.get("LLM_API_BASE", "https://api.openai.com/v1"),
        help="Base URL of any OpenAI-compatible API (env: LLM_API_BASE)",
    )

    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=500,
        help="Maximum generated answer tokens. Default: 500",
    )

    # ============================================================
    # Context budget
    # ============================================================

    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=2000,
        help="Maximum total context tokens. Default: 2000",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> RAGConfig:
    """Create RAGConfig from CLI args."""

    api_key = load_api_key(args.apikey_txt)

    if args.config:
        config = RAGConfig.from_json(args.config)
        config.corpus_path = str(Path(args.corpus_dir).resolve())
        config.llm_api_key = api_key
        config.llm_api_base = args.llm_api_base
        config.llm_model = args.generation_model
        return config

    return RAGConfig(
        corpus_path=args.corpus_dir,

        # Chunking
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_parent_child_chunking=args.use_parent_child_chunking,
        parent_max_tokens=args.parent_max_tokens,

        # Embeddings
        embedding_model=args.embedding_model,
        normalize_embeddings_for_faiss=args.normalize_embeddings_for_faiss,

        # Retrieval
        retriever_type=args.retriever_type,
        top_k=args.top_k,
        retrieval_pool_k=args.retrieval_pool_k,
        use_multi_query_retrieval=args.use_multi_query_retrieval,
        multi_query_count=args.multi_query_count,
        hybrid_alpha=args.hybrid_alpha,
        hybrid_candidate_k=args.hybrid_candidate_k,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        use_retrieval_threshold=args.use_retrieval_threshold,
        retrieval_score_threshold=args.retrieval_score_threshold,
        min_retrieved=args.min_retrieved,
        max_retrieved=args.max_retrieved,

        # Reranking
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,

        # LLM
        llm_model=args.generation_model,
        llm_api_base=args.llm_api_base,
        llm_api_key=api_key,
        llm_max_tokens=args.llm_max_tokens,

        # Context budget
        max_context_tokens=args.max_context_tokens,
    )


def main():
    """Main CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not Path(args.corpus_dir).exists():
        logger.error(f"Corpus directory not found: {args.corpus_dir}")
        sys.exit(1)

    try:
        logger.info("=" * 80)
        logger.info("RAG PIPELINE EXECUTION")
        logger.info("=" * 80)

        Path("logs").mkdir(parents=True, exist_ok=True)

        retrieval_log_path = Path("logs/retrieved_chunks.jsonl")
        if retrieval_log_path.exists():
            retrieval_log_path.unlink()

        questions_data = load_input_file(args.input)

        config = config_from_args(args)

        logger.info(f"Config: {config.to_dict()}")

        logger.info("\nBuilding RAG pipeline...")

        pipeline = create_pipeline(config)
        pipeline.build()

        logger.info(f"\nProcessing {len(questions_data)} questions...")

        results = []

        for item in questions_data:
            question_id = item.get("question_id")
            question = item.get("question")

            if not question:
                logger.warning(
                    f"Skipping item {question_id}: no question text"
                )
                continue

            logger.info(
                f"\nQuestion {question_id}: {question[:100]}..."
            )

            result = pipeline.query(question)

            output_item = {
                "question_id": question_id,
                "answer": result.get("answer", ""),
                "sources": [
                    {
                        "file": source.get("filename", "unknown"),
                        "lines": [
                            source.get("start_line", 0),
                            source.get("end_line", 0),
                        ],
                    }
                    for source in result.get("sources", [])
                ],
                "retrieved_context": result.get("retrieved_context", ""),
            }

            results.append(output_item)

        logger.info(f"\nSaving {len(results)} results...")
        save_output_file(results, args.output)

        logger.info("=" * 80)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()