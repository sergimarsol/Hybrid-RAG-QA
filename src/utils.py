"""Utility functions for RAG pipeline."""

import logging
from typing import List, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Rough heuristic: 1 token ≈ 4 characters for English text.
    """
    return len(text) // 4


def truncate_to_token_budget(text: str, max_tokens: int) -> Tuple[str, int]:
    """
    Truncate text to stay within token budget.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        
    Returns:
        Tuple of (truncated_text, actual_tokens)
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text, estimate_tokens(text)
    else:
        truncated = text[:max_chars]
        # Try to truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated, estimate_tokens(truncated)


def calculate_context_usage(
    system_prompt: str,
    question: str,
    retrieved_chunks: List[str]
) -> Tuple[int, List[int]]:
    """
    Calculate token usage for context assembly.
    
    Args:
        system_prompt: System prompt text
        question: User question
        retrieved_chunks: List of retrieved document chunks
        
    Returns:
        Tuple of (total_tokens, chunk_tokens)
    """
    system_tokens = estimate_tokens(system_prompt)
    question_tokens = estimate_tokens(question)
    
    chunk_tokens = [estimate_tokens(chunk) for chunk in retrieved_chunks]
    
    total = system_tokens + question_tokens + sum(chunk_tokens)
    
    return total, chunk_tokens


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
