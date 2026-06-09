"""Similarity search over the Qdrant paper-chunk index (M5.2).

The retrieval half of the vector-RAG baseline: embed a natural-language query
with bge-small and return the nearest chunks (with their payload + score).

bge asymmetric retrieval: the *query* gets an instruction prefix, the indexed
passages do not (they were embedded prefix-free in M5.1). Skipping this prefix
measurably hurts recall, so it is applied here and only here. The query vector
is L2-normalized to match the indexed vectors under cosine.

    python src/vector/search.py "who works on retrieval augmented generation?"
    python src/vector/search.py --k 10 "multi-hop reasoning over knowledge graphs"

Used by: vector_rag baseline (M5.3), hybrid router (M7).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# bge-small-en-v1.5 retrieval instruction (queries only; passages are prefix-free).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def _get_model(model_name: str):
    """Lazily load and cache the embedding model (one load per process)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
    return _model


def embed_query(query: str, model_name: str = config.EMBEDDING_MODEL) -> list[float]:
    """Embed a query (with bge prefix), L2-normalized, as a plain float list."""
    model = _get_model(model_name)
    vec = model.encode(
        [QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    return vec.tolist()


def search(
    query: str,
    k: int = 5,
    client=None,
    collection: str = config.QDRANT_COLLECTION,
) -> list[dict]:
    """Return the top-k chunks for `query`: each is its payload + a `score`."""
    if client is None:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=config.QDRANT_URL)
    qvec = embed_query(query)
    res = client.query_points(
        collection_name=collection, query=qvec, limit=k, with_payload=True
    )
    return [{"score": pt.score, **(pt.payload or {})} for pt in res.points]


def main() -> int:
    args = sys.argv[1:]
    k = 5
    if "--k" in args:
        i = args.index("--k")
        k = int(args[i + 1])
        del args[i : i + 2]
    if not args:
        print('usage: python src/vector/search.py [--k N] "your question"', file=sys.stderr)
        return 1
    query = " ".join(args)

    hits = search(query, k=k)
    print(f"Query: {query}\nTop {len(hits)}:")
    for rank, h in enumerate(hits, 1):
        title = h.get("title", "")
        print(f"  {rank}. [{h['score']:.4f}] {title}  ({h.get('paper_id')}#{h.get('chunk_index')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
