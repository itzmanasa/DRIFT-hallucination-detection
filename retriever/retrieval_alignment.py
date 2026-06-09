import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model once at module level
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.
    Returns value between 0 and 1.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Embeds all chunks once and returns embedding matrix.
    Call this once per query, not once per span.
    
    Args:
        chunks: list of text chunks from D
    
    Returns:
        numpy array of shape (num_chunks, embedding_dim)
    """
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return embeddings


def get_top_k_chunks(
    span: str,
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    k: int = 3
) -> list[str]:
    """
    Transformation 2 — Retrieval Alignment.
    Finds the top-k most semantically similar chunks to a DriftSpan.
    
    Args:
        span: one DriftSpan string (si)
        chunks: list of all retrieved chunks (D)
        chunk_embeddings: precomputed embeddings for all chunks
        k: number of top chunks to return (default 3)
    
    Returns:
        Di — list of top-k most relevant chunks for this span
    """
    # Embed the span
    span_embedding = embedder.encode(span, convert_to_numpy=True)
    
    # Compute cosine similarity between span and every chunk
    similarities = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(span_embedding, chunk_emb)
        similarities.append((i, sim))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k chunks
    top_k_indices = [idx for idx, sim in similarities[:k]]
    top_k_chunks = [chunks[idx] for idx in top_k_indices]
    
    return top_k_chunks


if __name__ == "__main__":
    # Test with example data
    test_spans = [
        "The drug was approved by FDA in 2022",
        "The drug had no serious side effects"
    ]
    
    test_chunks = [
        "The FDA granted approval to the medication in late 2022 after reviewing trial data.",
        "Manufacturing of the drug began in 2021 at facilities in New Jersey.",
        "Clinical trials reported minimal adverse events with no serious complications observed.",
        "The drug showed strong efficacy in reducing fever symptoms across age groups.",
        "Regulatory bodies in Europe are still reviewing the drug for approval."
    ]
    
    print("Precomputing chunk embeddings...")
    chunk_embeddings = embed_chunks(test_chunks)
    print(f"Embedded {len(test_chunks)} chunks\n")
    
    for span in test_spans:
        print(f"DriftSpan: {span}")
        di = get_top_k_chunks(span, test_chunks, chunk_embeddings, k=3)
        print(f"Top-3 relevant chunks (Di):")
        for j, chunk in enumerate(di):
            print(f"  chunk {j+1}: {chunk}")
        print()