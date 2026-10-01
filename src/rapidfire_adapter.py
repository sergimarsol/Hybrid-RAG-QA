"""Adapter to let RapidFire use the project's RAGPipeline as an inference engine.

This module exposes:
- `RapidFireRAGPipeline`: a minimal pipeline object RapidFire expects (get_engine_class/get_engine_kwargs/model_config)
- `RAGInferenceEngine`: an `InferenceEngine` subclass that runs the project's `RAGPipeline` on each prompt
- `PRECOMPUTED_RETRIEVALS`: module-level dict populated during generation so postprocess can access retrieved spans

The actor runs this engine remotely via Ray; the engine instantiates `RAGPipeline` and calls `query()`.
"""
from typing import Any, List
import hashlib
import json

from rapidfireai.evals.actors.inference_engines import InferenceEngine

from src.config import RAGConfig
from src.rag_pipeline import RAGPipeline

# Simple shared store to expose retrieval metadata to postprocess_fn running in the same actor process
PRECOMPUTED_RETRIEVALS: dict[int, list] = {}


class RAGInferenceEngine(InferenceEngine):
    """InferenceEngine that uses the project's RAGPipeline to answer prompts.

    Engine is constructed with `rag_config` (dict or RAGConfig). It builds the
    pipeline once and answers each prompt by calling `RAGPipeline.query()`.
    """

    def __init__(self, rag_config: Any):
        # Accept either dict or RAGConfig
        if isinstance(rag_config, dict):
            cfg = RAGConfig.from_dict(rag_config)
        elif isinstance(rag_config, RAGConfig):
            cfg = rag_config
        else:
            # Fallback: try passing kwargs to RAGConfig
            cfg = RAGConfig(**(rag_config or {}))

        # Build pipeline (this can be moderately expensive; done per actor)
        self.rag_pipeline = RAGPipeline(cfg)
        self.rag_pipeline.build()

    def generate(self, prompts: List[Any], **kwargs) -> List[str]:
        """Generate answers for a list of prompts.

        Each prompt may be a plain string or a list of message dicts (RapidFire/OpenAI style).
        We attempt to extract a user question string and run `RAGPipeline.query`.
        We also populate `PRECOMPUTED_RETRIEVALS` with the `sources` for postprocessing.
        """
        PRECOMPUTED_RETRIEVALS.clear()
        outputs: List[str] = []
        for i, prompt in enumerate(prompts):
            # Best-effort extraction of textual question
            if isinstance(prompt, str):
                question = prompt
            elif isinstance(prompt, list):
                # prompt is likely a list of message dicts
                question = None
                for msg in reversed(prompt):
                    if isinstance(msg, dict) and msg.get("content"):
                        question = msg["content"]
                        break
                if question is None:
                    question = str(prompt)
            else:
                question = str(prompt)

            result = self.rag_pipeline.query(question)

            # Save sources as normalized tuples for later evaluation
            sources = []
            for s in result.get("sources", []):
                # Each source is a dict with filename, start_line, end_line
                try:
                    sources.append((s.get("filename"), int(s.get("start_line", 0)), int(s.get("end_line", 0))))
                except Exception:
                    # best-effort fallback
                    sources.append(tuple(s.values()))

            PRECOMPUTED_RETRIEVALS[i] = sources
            outputs.append(result.get("answer", ""))

        return outputs

    def cleanup(self):
        # Nothing special to clean up for the project's pipeline
        return None


class RapidFireRAGPipeline:
    """Minimal pipeline object RapidFire scheduler expects.

    RapidFire will call `get_engine_class()` and `get_engine_kwargs()` to
    initialize the engine on actors.
    """

    def __init__(self, rag_config: Any):
        # Store either RAGConfig or its dict; keep light-weight for pickling
        if isinstance(rag_config, RAGConfig):
            self._rag_config = rag_config.to_dict()
        elif isinstance(rag_config, dict):
            self._rag_config = rag_config
        else:
            # Try to coerce
            try:
                self._rag_config = rag_config.to_dict()
            except Exception:
                self._rag_config = {}

        # Expose a small model_config so RapidFire can name the pipeline
        self.model_config = {"model": self._rag_config.get("llm_model", "unknown")}

        # Provide a lightweight `rag` attribute so RapidFire can show retrieval knobs
        class _RAGStub:
            def __init__(self, cfg: dict):
                self.search_type = cfg.get("retriever_type", "similarity")
                # search_kwargs: include top_k if present
                self.search_kwargs = {
                    "k": cfg.get("top_k", 5),
                    "hybrid_alpha": cfg.get("hybrid_alpha", 0.5),
                    "hybrid_candidate_k": cfg.get("hybrid_candidate_k", 50),
                    "bm25_k1": cfg.get("bm25_k1", 1.5),
                    "bm25_b": cfg.get("bm25_b", 0.75),
                    "normalize_embeddings_for_faiss": cfg.get("normalize_embeddings_for_faiss", False),
                    "use_retrieval_threshold": cfg.get("use_retrieval_threshold", False),
                    "retrieval_score_threshold": cfg.get("retrieval_score_threshold", 0.5),
                    "min_retrieved": cfg.get("min_retrieved", 1),
                    "max_retrieved": cfg.get("max_retrieved", cfg.get("top_k", 5)),
                }
                # expose chunk size/overlap for display
                self.chunk_size = cfg.get("chunk_size")
                self.chunk_overlap = cfg.get("chunk_overlap")
                # embedding kwargs for display
                self.embedding_kwargs = {"model_name": cfg.get("embedding_model")}
                # flags used by display
                self.enable_gpu_search = False
                # keep a copy of the cfg for hashing/serializing
                try:
                    # ensure keys are JSON-serializable
                    self._cfg = dict(cfg) if cfg is not None else {}
                except Exception:
                    self._cfg = {}

            def get_hash(self) -> str:
                """Return a deterministic hash for this RAG spec for RapidFire's deduping.

                The controller uses `get_hash()` to compute unique context ids across
                pipeline configs. We provide a stable SHA1 over the JSON-serialized
                config (sorted keys).
                """
                try:
                    s = json.dumps(self._cfg, sort_keys=True, default=str)
                except Exception:
                    s = str(self._cfg)
                return hashlib.sha1(s.encode("utf-8")).hexdigest()
                
            # RapidFire may access additional attributes on the rag spec
            # such as `vector_store_cfg` and `reranker_cfg`. Provide
            # minimal sensible defaults so the controller can inspect them
            # without raising AttributeError.
            @property
            def vector_store_cfg(self):
                # Controller expects a dict with at least 'type' and 'connection'
                return self._cfg.get("vector_store_cfg", {"type": "FAISS (default, CPU)", "connection": None})

            @property
            def reranker_cfg(self):
                # Some RapidFire code paths expect a reranker config; return None
                return self._cfg.get("reranker_cfg", None)
            
            @property
            def reranker_cls(self):
                # Some controller code checks for a reranker class; return None
                return self._cfg.get("reranker_cls", None)
            
            @property
            def reranker_kwargs(self):
                # Older RapidFire internals may expect `reranker_kwargs` naming.
                return self._cfg.get("reranker_kwargs", {})

            def build_pipeline(self):
                """Minimal no-op build to satisfy RapidFire's DocProcessingActor.

                We do not construct a full LangChainRagSpec here — the actual
                retrieval+generation is performed inside the inference engine.
                This method sets lightweight attributes the controller expects
                so the context build step can complete without raising.
                """
                # Embedding configuration (class or identifier + kwargs)
                self.embedding_cls = self._cfg.get("embedding_cls", None)
                self.embedding_kwargs = self._cfg.get("embedding_kwargs", {"model_name": self._cfg.get("embedding_model")})
                # Template used for context formatting (optional)
                self.template = self._cfg.get("template", None)
                # Indicate whether GPU search is intended
                self.enable_gpu_search = self._cfg.get("enable_gpu_search", False)
                # We do not build a shared vector store here; engine will handle retrieval
                self.vector_store = None
                self.retriever = None
                # No-op return
                return None

        self.rag = _RAGStub(self._rag_config)

    def get_engine_class(self):
        return RAGInferenceEngine

    def get_engine_kwargs(self) -> dict:
        return {"rag_config": self._rag_config}

    # No rag or prompt_manager attributes (we perform retrieval+generation inside engine)


__all__ = ["RAGInferenceEngine", "RapidFireRAGPipeline", "PRECOMPUTED_RETRIEVALS"]
