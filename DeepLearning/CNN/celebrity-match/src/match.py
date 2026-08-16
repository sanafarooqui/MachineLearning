"""Match a query selfie against the FAISS index of celebrity embeddings."""

import argparse
import json
from pathlib import Path

# torch must be imported before faiss: importing faiss first corrupts torch's
# OpenMP thread pool and segfaults on macOS when the FaceNet weights load.
import torch
from facenet_pytorch import InceptionResnetV1
import faiss
import numpy as np

from preprocess import preprocess_image

EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "index.faiss"
LABELS_PATH = EMBEDDINGS_DIR / "labels.json"

# Same frozen, pretrained model used at index-build time — matching the
# indexing pipeline exactly is what makes the embeddings comparable.
_facenet = InceptionResnetV1(pretrained="vggface2").eval()


def embed_face(face: torch.Tensor) -> np.ndarray:
    """Run a preprocessed face tensor through FaceNet and return a unit-normalized embedding.

    Args:
        face: A (3, 160, 160) preprocessed face tensor.

    Returns:
        A unit-length 512-dim embedding as a numpy array.
    """
    # Duplicated from build_index.py rather than imported, so match.py (and
    # the API that wraps it) can run standalone without depending on the
    # batch-indexing script.
    with torch.no_grad():
        embedding = _facenet(face.unsqueeze(0)).squeeze(0).numpy()
    return embedding / np.linalg.norm(embedding)


def match(image_path: Path, top_k: int = 5) -> list[dict]:
    """Find the top-k closest celebrity matches for a query selfie.

    Args:
        image_path: Path to the query selfie.
        top_k: Number of matches to return.

    Returns:
        A list of {celebrity_name, similarity_score} dicts, best match first.
    """
    # Loaded fresh on every call — fine for CLI use and low-traffic serving.
    # A long-lived, high-QPS server should cache these instead of re-reading
    # from disk per request.
    index = faiss.read_index(str(INDEX_PATH))
    labels = json.loads(LABELS_PATH.read_text())

    face = preprocess_image(image_path)
    # FAISS expects a 2D (n_queries, dim) float32 array, even for one query.
    query_embedding = embed_face(face).astype("float32").reshape(1, -1)

    # Both index and query are unit-normalized, so inner product here is
    # equivalent to cosine similarity.
    scores, indices = index.search(query_embedding, min(top_k, len(labels)))

    # scores/indices come back as (1, top_k) arrays (one row per query);
    # [0] unwraps that single row. FAISS gives us row indices into the
    # index, so labels[idx] maps each back to a celebrity name.
    return [
        {"celebrity_name": labels[idx], "similarity_score": float(score)}
        for score, idx in zip(scores[0], indices[0])
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match a selfie to the most similar celebrity.")
    parser.add_argument("image_path", type=Path, help="Path to the query selfie image.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of matches to return.")
    parser.add_argument("--verbose", action="store_true", help="Show top-10 results instead of top-5.")
    args = parser.parse_args()

    # --top-k always wins if given explicitly; --verbose is just a shorthand
    # default of 10 for eyeballing more candidates during manual testing.
    top_k = args.top_k if args.top_k is not None else (10 if args.verbose else 5)
    results = match(args.image_path, top_k=top_k)

    print(f"Top {len(results)} matches for {args.image_path}:")
    for rank, result in enumerate(results, start=1):
        print(f"  {rank}. {result['celebrity_name']:<25} similarity: {result['similarity_score']:.4f}")
