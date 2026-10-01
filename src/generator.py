"""LLM generation component for RAG pipeline."""

import json
import os
import re
from typing import List, Optional
from pathlib import Path
import openai

from .utils import get_logger, estimate_tokens

logger = get_logger(__name__)


class ContextAssembler:
    """Assemble context within token budget."""
    
    def __init__(
        self,
        system_prompt: str,
        max_context_tokens: int = 2000,
        include_metadata: bool = True
    ):
        """
        Initialize context assembler.
        
        Args:
            system_prompt: System prompt text
            max_context_tokens: Maximum tokens for entire context
            include_metadata: Include chunk metadata in context
        """
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.include_metadata = include_metadata
    
    def assemble(
        self,
        question: str,
        retrieved_chunks: List[str],
        chunk_metadata: Optional[List[dict]] = None,
    ) -> tuple[str, dict]:
        """
        Assemble final prompt context while enforcing the full 2000-token budget.

        The budget includes:
        - system prompt
        - source material
        - source labels / metadata
        - question
        - wrapper text like "System:", "Source Material:", "Question:"

        Returns:
            (final_context, token_usage)
        """
        # Start with the fixed shell of the prompt
        base_context = (
            f"System: {self.system_prompt}\n\n"
            f"Source Material:\n\n"
            f"Question: {question}"
        )

        base_tokens = estimate_tokens(base_context)
        system_tokens = estimate_tokens(self.system_prompt)
        question_tokens = estimate_tokens(question)

        if base_tokens >= self.max_context_tokens:
            logger.warning(
                f"Base prompt already uses {base_tokens} tokens, "
                f"leaving no room for retrieved chunks."
            )
            return base_context, {
                "system_tokens": system_tokens,
                "question_tokens": question_tokens,
                "content_tokens": 0,
                "total_tokens": base_tokens,
                "chunks_used": 0,
                "total_chunks_available": len(retrieved_chunks),
            }

        context_parts = []
        used_chunks = 0

        for i, chunk in enumerate(retrieved_chunks):
            chunk_block = chunk

            if self.include_metadata and chunk_metadata and i < len(chunk_metadata):
                filename = chunk_metadata[i].get("filename", "unknown")
                chunk_block = f"[Source: {filename}]\n{chunk}"

            candidate_parts = context_parts + [chunk_block]
            candidate_body = "\n\n".join(candidate_parts)

            candidate_context = (
                f"System: {self.system_prompt}\n\n"
                f"Source Material:\n{candidate_body}\n\n"
                f"Question: {question}"
            )

            candidate_tokens = estimate_tokens(candidate_context)

            if candidate_tokens > self.max_context_tokens:
                logger.info(f"Reached token budget after {used_chunks} chunks")
                break

            context_parts = candidate_parts
            used_chunks += 1

        context_body = "\n\n".join(context_parts)

        final_context = (
            f"System: {self.system_prompt}\n\n"
            f"Source Material:\n{context_body}\n\n"
            f"Question: {question}"
        )

        total_tokens = estimate_tokens(final_context)
        content_tokens = max(0, total_tokens - system_tokens - question_tokens)

        return final_context, {
            "system_tokens": system_tokens,
            "question_tokens": question_tokens,
            "content_tokens": content_tokens,
            "total_tokens": total_tokens,
            "chunks_used": used_chunks,
            "total_chunks_available": len(retrieved_chunks),
        }


class SimpleGenerator:
    """Simple LLM generator supporting mock mode and Triton chat completions."""
    
    def __init__(self, model_name: str = "mock", api_base: str | None = None, api_key: str | None = None):
        """Initialize generator."""
        self.model_name = model_name
        self.is_mock = "mock" in model_name.lower()
        self.api_base = api_base or os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or self._load_api_key()
        self.client = None
        
        if not self.is_mock:
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                )

                logger.info(
                    f"Initialized LLM client for model={self.model_name}, "
                    f"api_base={self.api_base}"
                )

            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
                self.is_mock = True

    def _load_api_key(self) -> str | None:
        """
        Load API key from ~/api-key.txt.

        Returns:
            API key string if found, else None
        """
        key_path = Path.home() / "api-key.txt"
    
        if key_path.exists():
            return key_path.read_text(encoding="utf-8").strip()
    
        logger.warning("api-key.txt not found in home directory")
        return None
    
    def generate(self, context: str, max_tokens: int = 500) -> str:
        """
        Generate answer from context.
        
        Args:
            context: Assembled context with question and source material
            max_tokens: Max tokens in response
            
        Returns:
            Generated answer
        """
        if self.is_mock:
            return self._generate_mock(context)
        else:
            return self._generate_openai(context, max_tokens)

    def generate_query_variants(self, question: str, num_queries: int = 3) -> List[str]:
        """Generate complementary retrieval queries for the given question."""
        if num_queries <= 0:
            return []

        if self.is_mock:
            return [question]

        if self.client is None:
            raise RuntimeError("LLM client is not initialized")

        prompt = (
            "You are generating retrieval search queries for a RAG documentation search system. "
            "Return only a JSON array of strings, no markdown, no commentary. "
            f"Produce exactly {num_queries} search queries.\n\n"

            "The goal is to retrieve complementary evidence from different documents or sections, "
            "not just paraphrase the question.\n\n"

            "Rules:\n"
            "- Preserve all named entities, technical terms, APIs, class names, dates, numbers, and constraints.\n"
            "- Do not answer the question.\n"
            "- Do not broaden the scope of the question.\n"
            "- Each query should target a different evidence angle or wording.\n"
            "- Prefer queries that could retrieve complementary chunks from different documents.\n"
            "- Include implementation terms, related mechanisms, dependencies, configuration terms, "
            "or component names when relevant.\n"
            "- Avoid vague paraphrases.\n"
            "- Avoid adding facts not present in the original question.\n\n"

            "Possible query strategies include:\n"
            "1. Direct answer phrasing\n"
            "2. Background or definition retrieval\n"
            "3. Related implementation/component terminology\n"
            "4. Error handling, constraints, or edge cases\n"
            "5. Alternative API or class naming conventions\n\n"

            f"Question: {question}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You generate concise retrieval queries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=256,
            )

            content = response.choices[0].message.content
            if content is None:
                return [question]

            content = content.strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
            parsed = json.loads(content)

            if isinstance(parsed, dict):
                parsed = parsed.get("queries", parsed.get("items", []))

            if not isinstance(parsed, list):
                return [question]

            variants: List[str] = []
            seen = {question.strip().lower()}

            for item in parsed:
                if not isinstance(item, str):
                    continue

                variant = item.strip()
                if not variant:
                    continue

                normalized = variant.lower()
                if normalized in seen:
                    continue

                seen.add(normalized)
                variants.append(variant)

            return variants[:num_queries]

        except Exception as e:
            logger.warning(f"Failed to generate query variants: {e}")
            return [question]
    
    def _generate_mock(self, context: str) -> str:
        """Generate mock response (for testing)."""
        # Simple mock: extract question and return a generic response
        if "Question:" in context:
            question = context.split("Question:")[-1].strip()[:100]
            return f"Based on the documentation, regarding '{question}': [This is a mock response for testing. In production, this would use an LLM.]"
        return "This is a mock response. An LLM would generate a proper answer based on the context."
    
    def _generate_openai(self, context: str, max_tokens: int = 500) -> str:
        """
        Generate response using Triton/OpenAI-compatible chat completions.

        Args:
            context: Prompt/context sent to the LLM
            max_tokens: Maximum number of output tokens

        Returns:
            Model response text
        """
        if self.client is None:
            raise RuntimeError("LLM client is not initialized")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": context}
                ],
                max_tokens=max_tokens
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            logger.error(f"Error calling LLM API: {e}", exc_info=True)
            return f"Error generating answer: {str(e)}"

def create_generator(model_name: str = "mock", api_base: str | None = None, api_key: str | None = None) -> SimpleGenerator:
    """Create a generator instance."""
    return SimpleGenerator(model_name=model_name, api_base=api_base, api_key=api_key)