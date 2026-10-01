#!/usr/bin/env python3
"""
Generation Evaluation Script
Evaluates generated answers using an LLM judge.

Metrics:
- correctness (0 or 1)
- faithfulness (0 or 1)
- completeness (0-5)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.project1_eval import call_judge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_json(filepath: str):
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_generation(
    golden_qa: List[Dict],
    generated_answers: List[Dict],
    judge_model: str = "mock",
    judge_base_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
) -> Dict:
    """Evaluate generated answers against golden answers with an LLM judge."""
    golden_by_id = {item["question_id"]: item for item in golden_qa}
    generated_by_id = {item["question_id"]: item for item in generated_answers}

    total_correctness = 0.0
    total_completeness = 0.0
    total_faithfulness = 0.0
    total_overall = 0.0
    questions_evaluated = 0
    per_question_results = []
    judge_failures = 0

    for qid, gold in golden_by_id.items():
        if qid not in generated_by_id:
            logger.warning(f"Question {qid} missing from generated answers")
            continue

        pred = generated_by_id[qid]

        question = gold["question"]
        reference_answer = gold["answer"]
        predicted_answer = pred.get("answer", "")

        retrieved_context = pred.get(
            "retrieved_context",
            pred.get("context", pred.get("contexts", ""))
        )

        if isinstance(retrieved_context, list):
            retrieved_context = "\n\n".join(str(item) for item in retrieved_context)

        # Use official call_judge from project1_eval
        judge_result = call_judge(
            query=question,
            reference_answer=reference_answer,
            retrieved_context=retrieved_context,
            system_answer=predicted_answer,
            model=judge_model,
            base_url=judge_base_url,
            api_key=judge_api_key,
        )

        if judge_result.get("failed"):
            judge_failures += 1
            logger.warning(f"Judge failed for question {qid}: {judge_result.get('error', 'unknown')}")
            # On failure, scores are 0/0/0 (silent failure hurts, not inflates)
            correctness = 0.0
            faithfulness = 0.0
            completeness = 0.0
            reasoning = f"Judge failed: {judge_result.get('error', 'unknown')}"
        else:
            correctness = judge_result["correctness"]
            faithfulness = judge_result["faithfulness"]
            completeness = judge_result["completeness"]
            reasoning = ""

        overall = (correctness + faithfulness + completeness) / 7.0

        total_correctness += correctness
        total_completeness += completeness
        total_faithfulness += faithfulness
        total_overall += overall
        questions_evaluated += 1

        per_question_results.append({
            "question_id": qid,
            "question": question,
            "correctness": correctness,
            "completeness": completeness,
            "faithfulness": faithfulness,
            "overall": overall,
            "reasoning": reasoning,
        })

    if questions_evaluated == 0:
        return {
            "questions_evaluated": 0,
            "avg_correctness": 0.0,
            "avg_completeness": 0.0,
            "avg_faithfulness": 0.0,
            "avg_overall": 0.0,
            "judge_failures": 0,
            "per_question_results": [],
        }

    return {
        "questions_evaluated": questions_evaluated,
        "avg_correctness": total_correctness / questions_evaluated,
        "avg_completeness": total_completeness / questions_evaluated,
        "avg_faithfulness": total_faithfulness / questions_evaluated,
        "avg_overall": total_overall / questions_evaluated,
        "judge_failures": judge_failures,
        "per_question_results": per_question_results,
    }


def print_results(results: Dict, output_file: Optional[str] = None):
    """Pretty-print generation eval results."""
    print("\n" + "=" * 80)
    print("GENERATION EVALUATION RESULTS")
    print("=" * 80)
    print(f"Questions evaluated: {results['questions_evaluated']}")
    judge_failures = results.get('judge_failures', 0)
    if judge_failures > 0:
        print(f"⚠️  Judge failures: {judge_failures}")
    print(f"Average Correctness:  {results['avg_correctness']:.3f} / 1")
    print(f"Average Completeness: {results['avg_completeness']:.3f} / 5")
    print(f"Average Faithfulness: {results['avg_faithfulness']:.3f} / 1")
    print(f"Average Overall:      {results['avg_overall']:.3f} / 1")

    print("\nPer-Question Results:")
    print(f"{'ID':<4} {'Corr':<6} {'Comp':<6} {'Faith':<6} {'Overall':<8} {'Question':<50}")
    print("-" * 100)

    for qr in results["per_question_results"]:
        question = qr["question"][:47] + "..." if len(qr["question"]) > 50 else qr["question"]
        print(
            f"{qr['question_id']:<4} "
            f"{qr['correctness']:<6.1f} "
            f"{qr['completeness']:<6.1f} "
            f"{qr['faithfulness']:<6.1f} "
            f"{qr['overall']:<8.2f} "
            f"{question:<50}"
        )

    print("=" * 80)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved generation eval results to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate generation quality with an LLM judge")
    parser.add_argument(
        "--golden",
        type=str,
        default="data/golden_qa_full.json",
        help="Path to golden Q&A JSON"
    )
    parser.add_argument(
        "--generated",
        type=str,
        default="data/generated_results.json",
        help="Path to generated answers JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/generation_eval_results.json",
        help="Path to save generation eval results"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="mock",
        help='Judge model name (official: claude-sonnet-4-6, or "mock" for testing)'
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help="Base URL for Triton/OpenAI-compatible API (default: env JUDGE_BASE_URL)"
    )

    args = parser.parse_args()

    if not Path(args.generated).exists():
        logger.error(f"Generated results file not found: {args.generated}")
        logger.info("First run main.py to produce generated answers.")
        return

    golden_qa = load_json(args.golden)
    generated_answers = load_json(args.generated)

    # Get API key from environment or file
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("JUDGE_API_KEY")
    if not api_key:
        key_path = Path.home() / "api-key.txt"
        if key_path.exists():
            api_key = key_path.read_text(encoding="utf-8").strip()

    results = evaluate_generation(
        golden_qa=golden_qa,
        generated_answers=generated_answers,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=api_key,
    )

    print_results(results, args.output)


if __name__ == "__main__":
    main()