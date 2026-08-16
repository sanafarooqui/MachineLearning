"""FastAPI backend wrapping the match.py pipeline behind an HTTP API."""

import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# src/ isn't an installed package, so we add it to the import path directly
# rather than restructuring the ML pipeline just to satisfy the API layer.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from match import match  # noqa: E402  (import must follow sys.path.insert above)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Celebrity Face Match API")

# Restricted to the local Next.js dev server. Widen this (or make it
# env-driven) before deploying the frontend to any other origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/match")
async def match_endpoint(file: UploadFile = File(...)) -> dict:
    """Accept an uploaded selfie and return the top-5 celebrity matches.

    Args:
        file: Uploaded image (jpeg/png/webp).

    Returns:
        {"matches": [{"celebrity_name": ..., "similarity_score": ...}, ...]}
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    # match()/preprocess_image() expect a filesystem path, not raw bytes, so
    # the upload is written to a temp file first. The `with` block deletes
    # it automatically once we're done, whether or not matching succeeds.
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()

        try:
            results = match(Path(tmp.name), top_k=5)
        except ValueError as exc:
            # Raised by preprocess_image() for "no face detected" or
            # "unreadable image" — a client error (422), not a server error.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"matches": results}
