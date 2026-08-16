"""Build a FAISS index of averaged FaceNet embeddings, one per celebrity identity."""

import json
from pathlib import Path

# torch must be imported before faiss: importing faiss first corrupts torch's
# OpenMP thread pool and segfaults on macOS when the FaceNet weights load.
import torch
from facenet_pytorch import InceptionResnetV1
import faiss
import numpy as np
from tqdm import tqdm

from preprocess import preprocess_image

# All paths are resolved relative to this file, so the script works no
# matter what directory it's invoked from.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "celebrities"
EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "index.faiss"
LABELS_PATH = EMBEDDINGS_DIR / "labels.json"

# Frozen, pretrained on VGGFace2 — no fine-tuning in Phase 1. Loaded once
# at import time and reused across every embed_face() call. .eval() turns
# off dropout/batchnorm training behavior since we're only doing inference.
_facenet = InceptionResnetV1(pretrained="vggface2").eval()


def embed_face(face: torch.Tensor) -> np.ndarray:
    """Run a preprocessed face tensor through FaceNet and return a unit-normalized embedding.

    Args:
        face: A (3, 160, 160) preprocessed face tensor.

    Returns:
        A unit-length 512-dim embedding as a numpy array.
    """
    # no_grad() skips building the autograd graph — we're not training,
    # just running inference, so this saves memory and time.
    with torch.no_grad():
        # unsqueeze(0) adds a batch dimension (FaceNet expects a batch,
        # even of size 1); squeeze(0) removes it again from the output.
        embedding = _facenet(face.unsqueeze(0)).squeeze(0).numpy()
    # Unit-normalize so that FAISS's inner product later behaves like
    # cosine similarity.
    return embedding / np.linalg.norm(embedding)


def build_index() -> None:
    """Walk data/celebrities/, average FaceNet embeddings per identity, and save a FAISS index."""
    # One subdirectory = one identity; the folder name becomes the label.
    identity_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())

    labels = []       # labels[i] will correspond to row i of the FAISS index
    embeddings = []    # one averaged, unit-normalized vector per identity
    num_images_processed = 0
    num_images_failed = 0

    for identity_dir in tqdm(identity_dirs, desc="Identities"):
        identity_embeddings = []
        for image_path in identity_dir.glob("*"):
            try:
                face = preprocess_image(image_path)
                identity_embeddings.append(embed_face(face))
                num_images_processed += 1
            except (FileNotFoundError, ValueError):
                # Bad file, unreadable image, or no face detected — skip it
                # rather than aborting the whole build over one bad photo.
                num_images_failed += 1

        if not identity_embeddings:
            continue

        # Average every image of this person into a single "centroid"
        # embedding, then re-normalize. This centroid is what actually gets
        # indexed — it's more robust to any one odd/mislabeled photo than
        # indexing every image individually would be.
        avg_embedding = np.mean(identity_embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        embeddings.append(avg_embedding)
        labels.append(identity_dir.name)

    # Stack into a (num_identities, 512) matrix for FAISS.
    embeddings = np.stack(embeddings).astype("float32")
    # IndexFlatIP = exact, brute-force inner-product search. Fine at this
    # scale (tens of identities); would need an approximate index (IVF,
    # HNSW) if this grew into the thousands+.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    LABELS_PATH.write_text(json.dumps(labels, indent=2))

    print(f"Indexed {len(labels)} celebrities from {num_images_processed} images "
          f"({num_images_failed} failed/skipped)")


if __name__ == "__main__":
    build_index()
