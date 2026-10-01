"""Embedding models for RAG pipeline."""

from typing import List, Optional
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings

from .utils import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Wrapper for embedding models."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info(f"Embedding model loaded. Dimension: {self.get_embedding_dim()}")
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        # Get a sample embedding to determine dimension
        sample_embedding = self.embeddings.embed_query("test")
        return len(sample_embedding)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return np.array(self.embeddings.embed_query(query))
    
    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        """
        Embed multiple documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embeddings.embed_documents(documents)
        return [np.array(e) for e in embeddings]
    
    def embed_chunk_list(self, chunks: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of chunks in batches.
        
        Args:
            chunks: List of chunk texts
            batch_size: Batch size for processing
            
        Returns:
            Matrix of embeddings (num_chunks x embedding_dim)
        """
        logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}")
        
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_embeddings = self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % (batch_size * 5) == 0:
                logger.info(f"Embedded {i+batch_size}/{len(chunks)} chunks")
        
        logger.info(f"Embedding complete. Shape: ({len(all_embeddings)}, {len(all_embeddings[0])})")
        
        return np.array(all_embeddings)


def get_embedding_model(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> EmbeddingModel:
    """Get an embedding model instance."""
    return EmbeddingModel(model_name)
