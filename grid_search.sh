#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# The golden QA file doubles as pipeline input (main.py only reads question_id/question).
INPUT_FILE="data/golden_qa_full_v2_clean.json"
CORPUS_DIR="data/sourcedocs"
GOLDEN_FILE="data/golden_qa_full_v2_clean.json"
APIKEY_TXT="${APIKEY_TXT:-$HOME/api-key.txt}"
JUDGE_MODEL="claude-sonnet-4-6"
LLM_MODEL="api-gpt-oss-120b"
LLM_API_BASE="${LLM_API_BASE:-https://api.openai.com/v1}"
RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_POOL_K=15
RETRIEVAL_K=5
OUTPUT_ROOT="logs/grid_search"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Missing input file: $INPUT_FILE" >&2
  exit 1
fi

if [[ ! -f "$APIKEY_TXT" ]]; then
  echo "Missing API key file: $APIKEY_TXT (set APIKEY_TXT to override)" >&2
  exit 1
fi

if [[ ! -d "$CORPUS_DIR" ]]; then
  echo "Missing corpus directory: $CORPUS_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

INPUT_SUBSET_FILE="$OUTPUT_ROOT/first_10_input.json"
GOLDEN_SUBSET_FILE="$OUTPUT_ROOT/first_10_golden.json"

python3 - <<'PY'
import json
from pathlib import Path

input_path = Path("data/golden_qa_full_v2_clean.json")
golden_path = Path("data/golden_qa_full_v2_clean.json")
output_input = Path("logs/grid_search/first_10_input.json")
output_golden = Path("logs/grid_search/first_10_golden.json")

output_input.parent.mkdir(parents=True, exist_ok=True)

with input_path.open("r", encoding="utf-8") as f:
  input_data = json.load(f)

with golden_path.open("r", encoding="utf-8") as f:
  golden_data = json.load(f)

with output_input.open("w", encoding="utf-8") as f:
  json.dump(input_data[:10], f, indent=2)

with output_golden.open("w", encoding="utf-8") as f:
  json.dump(golden_data[:10], f, indent=2)
PY

chunk_sizes=(150 256)
top_ks=(5 8)
reranker_modes=(off on)
retriever_types=(similarity mmr)

for chunk_size in "${chunk_sizes[@]}"; do
  for top_k in "${top_ks[@]}"; do
    for retriever_type in "${retriever_types[@]}"; do
      for reranker_mode in "${reranker_modes[@]}"; do
        run_name="chunk_${chunk_size}_topk_${top_k}_retriever_${retriever_type}_reranker_${reranker_mode}"
      run_dir="$OUTPUT_ROOT/$run_name"
      raw_results="$run_dir/raw_results.json"
      retrieval_eval="$run_dir/retrieval_eval.json"
      generation_eval="$run_dir/generation_eval.json"
      main_log="$run_dir/main.log"
      retrieval_log="$run_dir/retrieval.log"
      generation_log="$run_dir/generation.log"

      mkdir -p "$run_dir"

      main_cmd=(
        python3 main.py
        --input "$INPUT_SUBSET_FILE"
        --output "$raw_results"
        --corpus-dir "$CORPUS_DIR"
        --apikey-txt "$APIKEY_TXT"
        --generation-model "$LLM_MODEL"
        --llm-api-base "$LLM_API_BASE"
        --chunk-size "$chunk_size"
        --top-k "$top_k"
        --retriever-type "$retriever_type"
      )

      if [[ "$reranker_mode" == "on" ]]; then
        main_cmd+=(
          --use-reranker
          --retrieval-pool-k "$RETRIEVAL_POOL_K"
          --reranker-model "$RERANKER_MODEL"
        )
      fi

      {
        echo "Running: ${main_cmd[*]}"
        "${main_cmd[@]}"
      } &> "$main_log"

      python3 evaluate_retrieval.py \
        --output "$raw_results" \
        --validation "$GOLDEN_SUBSET_FILE" \
        --k "$RETRIEVAL_K" \
        > "$retrieval_eval" \
        2> "$retrieval_log"

      python3 run_judge.py \
        --golden "$GOLDEN_SUBSET_FILE" \
        --generated "$raw_results" \
        --output "$generation_eval" \
        --judge-model "$JUDGE_MODEL" \
        --judge-base-url "$LLM_API_BASE" \
        > /dev/null \
        2> "$generation_log"

      echo "Finished $run_name"
      done
    done
  done
done
