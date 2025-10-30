"""Embedding utilities for memory system.

Provides text embedding functionality for similarity-based retrieval
using OpenAI embeddings or local models.

Inspired by:
- AlphaCode's code clustering approach
- PaperCoder's semantic paper matching
"""

import os
from typing import List, Optional, Union
import numpy as np
import hashlib
import pickle


class EmbeddingCache:
    """Cache for embeddings to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = "./embedding_cache"):
        """Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "embeddings.pkl")
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def _get_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Cached embedding or None
        """
        key = self._get_key(text)
        return self.cache.get(key)
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """Cache embedding for text.
        
        Args:
            text: Input text
            embedding: Embedding vector
        """
        key = self._get_key(text)
        self.cache[key] = embedding
        self._save_cache()


class EmbeddingGenerator:
    """Generate embeddings for text using OpenAI or local models.
    
    Supports multiple backends:
    - OpenAI text-embedding-3-small (recommended for production)
    - OpenAI text-embedding-ada-002 (legacy)
    - Local sentence transformers (for offline use)
    """
    
    def __init__(self, 
                 backend: str = "openai",
                 model: str = "text-embedding-3-small",
                 cache_dir: str = "./embedding_cache"):
        """Initialize embedding generator.
        
        Args:
            backend: Backend to use ('openai' or 'local')
            model: Model name
            cache_dir: Directory for caching embeddings
        """
        self.backend = backend
        self.model = model
        self.cache = EmbeddingCache(cache_dir)
        
        if backend == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif backend == "local":
            try:
                from sentence_transformers import SentenceTransformer
                self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        
        # Generate new embedding
        if self.backend == "openai":
            embedding = self._embed_openai(text)
        else:
            embedding = self._embed_local(text)
        
        # Cache and return
        self.cache.set(text, embedding)
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Check cache for all texts
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            if self.backend == "openai":
                new_embeddings = self._embed_openai_batch(uncached_texts)
            else:
                new_embeddings = self._embed_local_batch(uncached_texts)
            
            # Insert new embeddings and cache them
            for i, embedding in zip(uncached_indices, new_embeddings):
                embeddings[i] = embedding
                self.cache.set(texts[i], embedding)
        
        return embeddings
    
    def _embed_openai(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding)
    
    def _embed_openai_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch using OpenAI API."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [np.array(data.embedding) for data in response.data]
    
    def _embed_local(self, text: str) -> np.ndarray:
        """Generate embedding using local model."""
        return self.local_model.encode(text, convert_to_numpy=True)
    
    def _embed_local_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch using local model."""
        embeddings = self.local_model.encode(texts, convert_to_numpy=True)
        return [emb for emb in embeddings]


def embed_paper(paper_content: Union[str, dict], 
                embedding_generator: EmbeddingGenerator) -> np.ndarray:
    """Generate embedding for a paper.
    
    Extracts key sections (title, abstract, methodology) and creates
    a representative embedding.
    
    Args:
        paper_content: Paper content as JSON or string
        embedding_generator: Embedding generator instance
        
    Returns:
        Paper embedding vector
    """
    if isinstance(paper_content, dict):
        # Extract key sections
        sections = []
        
        # Title
        if 'title' in paper_content:
            sections.append(paper_content['title'])
        
        # Abstract
        if 'abstract' in paper_content:
            sections.append(paper_content['abstract'])
        elif 'sections' in paper_content:
            for section in paper_content['sections']:
                if section.get('heading', '').lower() in ['abstract', 'introduction']:
                    sections.append(section.get('text', ''))
        
        # Combine sections
        text = ' '.join(sections)
    else:
        text = str(paper_content)
    
    # Truncate if too long (OpenAI has 8191 token limit for embeddings)
    max_chars = 30000
    if len(text) > max_chars:
        text = text[:max_chars]
    
    return embedding_generator.embed_text(text)


def embed_code(code: str, 
               embedding_generator: EmbeddingGenerator,
               include_docstring: bool = True) -> np.ndarray:
    """Generate embedding for code.
    
    Args:
        code: Source code
        embedding_generator: Embedding generator instance
        include_docstring: Whether to prioritize docstrings/comments
        
    Returns:
        Code embedding vector
    """
    if include_docstring:
        # Extract docstrings and comments for better semantic matching
        import re
        
        # Extract Python docstrings
        docstrings = re.findall(r'"""(.*?)"""', code, re.DOTALL)
        docstrings.extend(re.findall(r"'''(.*?)'''", code, re.DOTALL))
        
        # Extract comments
        comments = re.findall(r'#\s*(.*?)$', code, re.MULTILINE)
        
        # Combine with code
        text = ' '.join(docstrings + comments) + ' ' + code
    else:
        text = code
    
    # Truncate if too long
    max_chars = 30000
    if len(text) > max_chars:
        text = text[:max_chars]
    
    return embedding_generator.embed_text(text)


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def find_most_similar(query_embedding: np.ndarray,
                     candidate_embeddings: List[np.ndarray],
                     top_k: int = 5) -> List[tuple]:
    """Find most similar embeddings to a query.
    
    Args:
        query_embedding: Query embedding vector
        candidate_embeddings: List of candidate embeddings
        top_k: Number of top results to return
        
    Returns:
        List of (index, similarity_score) tuples
    """
    similarities = []
    
    for i, candidate in enumerate(candidate_embeddings):
        similarity = compute_similarity(query_embedding, candidate)
        similarities.append((i, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]
