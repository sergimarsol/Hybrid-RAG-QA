# RapidFire RAG QA System

Retrieval-Augmented Generation (RAG) system for question answering over RapidFire AI OSS documentation. The project implements a modular and configurable retrieval pipeline with support for hybrid retrieval, parent-child chunking, reranking, multi-query retrieval, and RapidFire AI experimentation.

---

## Features

- Structure-aware RST document parsing
- Configurable chunk size and overlap
- Parent-child chunking
- FAISS dense retrieval
- BM25 lexical retrieval
- Hybrid dense + sparse retrieval
- MMR retrieval
- Threshold-based retrieval
- Cross-encoder reranking
- Multi-query retrieval
- Strict 2,000-token context budgeting
- Metadata-aware context serialization
- Retrieval logging
- RapidFire AI integration for large-scale experimentation

---

## Repository Structure

```text
src/
├── config.py              # Central configuration system
├── data_loader.py         # RST parsing and chunking
├── embedding.py           # Embedding model wrapper
├── retriever.py           # FAISS/BM25/hybrid retrieval
├── generator.py           # Context assembly + generation
├── rag_pipeline.py        # Main pipeline orchestration
├── rapidfire_adapter.py   # RapidFire integration
├── utils.py               # Utility functions

main.py                    # CLI entry point
project1_eval.py           # Evaluation utilities
```

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The project requires:
- Python 3.10+
- FAISS
- sentence-transformers
- LangChain
- OpenAI-compatible API access

---

## Running the Pipeline

Basic usage:

```bash
python main.py \
  --input data/questions.json \
  --output outputs/results.json
```

Example using the final default configuration:

```bash
python main.py \
  --input data/questions.json \
  --output outputs/final_results.json \
  --retriever-type hybrid \
  --chunk-size 512 \
  --chunk-overlap 200 \
  --top-k 2 \
  --hybrid-alpha 0.15
```

Example with reranking enabled:

```bash
python main.py \
  --input data/questions.json \
  --output outputs/reranked.json \
  --use-reranker \
  --retrieval-pool-k 30
```

Example with parent-child chunking:

```bash
python main.py \
  --input data/questions.json \
  --output outputs/parent_child.json \
  --use-parent-child-chunking
```

---

## Main Retrieval Options

| Option | Description |
|---|---|
| `--retriever-type` | `similarity`, `mmr`, or `hybrid` |
| `--top-k` | Final number of retrieved chunks |
| `--chunk-size` | Child chunk size |
| `--chunk-overlap` | Overlap between chunks |
| `--use-reranker` | Enable cross-encoder reranking |
| `--use-multi-query-retrieval` | Enable query expansion |
| `--use-parent-child-chunking` | Enable hierarchical chunking |
| `--hybrid-alpha` | Dense vs BM25 weighting |

---

## Evaluation

The project supports:
- Retrieval evaluation (Precision, Recall, F1, MRR, Hit Rate)
- LLM-as-a-judge generation evaluation
- Validation dataset evaluation
- Custom golden QA evaluation
- RapidFire AI experiment tracking

---

## Logging

Retrieved chunks and retrieval metadata are automatically logged to:

```text
logs/retrieved_chunks.jsonl
```

This includes:
- Retrieved spans
- Scores
- Retrieval configuration
- Parent-child context information
- Multi-query retrieval traces

---

## Final Default Configuration

The final default configuration uses:

- Hybrid BM25 + dense retrieval
- `top-k = 2`
- Chunk size = 512
- Chunk overlap = 200
- `sentence-transformers/all-MiniLM-L6-v2`
- No reranking
- No multi-query retrieval

This configuration provided the best balance between retrieval precision, recall, and answer quality.