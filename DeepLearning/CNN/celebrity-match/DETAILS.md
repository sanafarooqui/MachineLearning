# Architecture & File Reference

## Pipeline overview

```
                     ┌─────────────────────┐
  build_index.py     │ data/celebrities/    │
  (offline, once)    │   <name>/*.jpg       │
                     └──────────┬───────────┘
                                │  for each identity folder
                                ▼
                     preprocess.py (MTCNN)
                     detect → align → crop 160x160
                                │
                                ▼
                     FaceNet (InceptionResnetV1,
                     pretrained="vggface2")
                     → 512-dim embedding per image
                                │
                                ▼
                     average embeddings per identity
                     → unit-normalize
                                │
                                ▼
                     ┌─────────────────────┐
                     │ embeddings/          │
                     │   index.faiss        │  FAISS IndexFlatIP
                     │   labels.json        │  identity names, index-aligned
                     └──────────┬───────────┘
                                │
  match.py           query selfie             │  loaded at query time
  (online, per query)     │                    │
                           ▼                    │
                     preprocess.py (MTCNN)      │
                     detect → align → crop      │
                           │                    │
                           ▼                    │
                     FaceNet → 512-dim embedding│
                           │                    │
                           ▼                    ▼
                     unit-normalize  ──►  faiss index.search()
                                                │
                                                ▼
                                   top-k {celebrity_name, similarity_score}
```

Because both the index and every query are unit-normalized, FAISS's inner
product (`IndexFlatIP`) is equivalent to cosine similarity — scores are
bounded in roughly [-1, 1], and in practice good matches land in 0.5-1.0.

## Serving layer

```
  frontend/ (Next.js, :3000)              api/ (FastAPI, :8000)
  ┌────────────────────────┐              ┌────────────────────────┐
  │ page.tsx (server)       │              │ GET  /health            │
  │ upload-matcher.tsx      │  POST/match  │ POST /match             │
  │ (client, "use client")  │─────────────►│  - validate content-type│
  │   - file input + preview│  multipart   │  - write upload to a    │
  │   - fetch()             │  form-data   │    NamedTemporaryFile   │
  │   - renders ranked      │◄─────────────│  - call match() from    │
  │     match cards         │  JSON        │    src/match.py         │
  └────────────────────────┘              └────────────────────────┘
```

`upload-matcher.tsx` is a client component (needs `useState`/`onChange`);
`page.tsx` stays a server component and just renders it — standard
Next.js App Router split of interactive vs. static UI. The API base URL
is read from `NEXT_PUBLIC_API_URL` (`frontend/.env.local`, gitignored),
defaulting to `http://localhost:8000`.

## File-by-file

### `src/preprocess.py`
Single responsibility: turn an arbitrary image file into a FaceNet-ready
tensor.

- `preprocess_image(image_path) -> torch.Tensor`
- Uses `facenet_pytorch.MTCNN(image_size=160, margin=0, select_largest=True, post_process=True)`
  as a module-level singleton, so the detector is loaded once per process.
- `select_largest=True` handles the "multiple faces in frame" case by
  keeping the largest bounding box (closest/most prominent face) and
  discarding the rest.
- `post_process=True` applies MTCNN's built-in fixed image standardization,
  so the returned tensor is already normalized to roughly [-1, 1] — no
  separate normalization step is needed downstream.
- Failure modes are explicit exceptions rather than `None`/silent skips:
  - `FileNotFoundError` — path doesn't exist.
  - `ValueError` — file exists but isn't a readable image, or MTCNN found
    no face at all.
- Reused unchanged by both `build_index.py` and `match.py`.

### `src/build_index.py`
Offline/batch step. Run once (or whenever `data/celebrities/` changes) to
produce the searchable index.

- Walks each subdirectory of `data/celebrities/` — one subdirectory =
  one identity, folder name = label.
- For every image in an identity's folder: `preprocess_image()` →
  `embed_face()` (FaceNet forward pass, unit-normalized). Images that fail
  preprocessing (bad file, no face) are counted and skipped, not fatal.
- All per-image embeddings for an identity are averaged, then the average
  is re-normalized to unit length. This single "centroid" embedding is
  what actually gets indexed — it's more robust to any one bad/odd photo
  than indexing every image individually.
- Embeddings are stacked into a `(num_identities, 512)` float32 matrix and
  added to a `faiss.IndexFlatIP` (exact, brute-force inner-product search —
  fine at this scale; would swap for an approximate index like `IVF` or
  `HNSW` if the identity count grew into the thousands+).
- Writes two artifacts to `embeddings/`:
  - `index.faiss` — the FAISS index itself.
  - `labels.json` — ordered list of celebrity names; `labels[i]`
    corresponds to row `i` of the index, so FAISS result indices map
    straight back to names.
- Prints a summary line: identities indexed, images processed, images
  failed/skipped.

**Import order matters:** `torch` (and anything that imports it, like
`facenet_pytorch`) must be imported *before* `faiss`. Importing `faiss`
first corrupts torch's OpenMP thread pool on macOS and segfaults the
process the moment a torch model tries to load its weights. Both
`build_index.py` and `match.py` import in the safe order and comment why.

### `src/match.py`
Online/query step. Run per selfie.

- `embed_face()` is duplicated from `build_index.py` rather than imported,
  since `match.py` is meant to run standalone (e.g. later embedded in a
  FastAPI endpoint) without depending on the batch-indexing script.
- `match(image_path, top_k=5) -> list[dict]`:
  1. Loads `index.faiss` and `labels.json` from `embeddings/` (loaded
     fresh per call — fine for CLI use and low-QPS serving; a long-lived
     server should cache these instead of reopening per request).
  2. Preprocesses and embeds the query image the same way as indexing.
  3. `index.search()` returns the `top_k` nearest identity centroids by
     inner product.
  4. Zips FAISS's `(scores, indices)` output against `labels` into
     `{"celebrity_name": ..., "similarity_score": ...}` dicts, best match
     first.
- CLI entrypoint (`argparse`):
  - positional `image_path`
  - `--top-k N` (default 5)
  - `--verbose` — shorthand for `--top-k 10`, for eyeballing more
    candidates during manual testing. An explicit `--top-k` always wins
    over `--verbose`'s default.

### `data/celebrities/`
Source dataset (Kaggle "Hollywood Celebrity Face Recognition Dataset").
One folder per identity, raw images inside. Read-only input to
`build_index.py` — never written to by the pipeline.

### `embeddings/`
Generated output, not source-of-truth. Safe to delete and regenerate by
rerunning `build_index.py`. Both files must stay in sync (same identity
count, same order) since `match.py` maps FAISS row indices to
`labels.json` positionally.

### `tests/test_selfies/`
Manually curated holdout images for validating match quality —
deliberately *not* sourced from `data/celebrities/`, so a strong result
here reflects real generalization rather than the model recognizing an
image it already averaged into the index. Current set was pulled from
Wikimedia Commons via Wikipedia's page-summary API (CC-licensed lead
photos), one per identity, to sanity-check top-1 accuracy end-to-end.

### `requirements.txt`
`facenet-pytorch`, `torch`, `torchvision`, `faiss-cpu`, `Pillow`, `numpy`,
`opencv-python`, `tqdm`, plus `fastapi`, `uvicorn[standard]`,
`python-multipart` for the API layer. CPU-only stack by design (Phase 1
constraint — no GPU required).

### `api/main.py`
FastAPI app wrapping `src/match.py` — the only new logic here is HTTP
plumbing, not ML.

- Inserts `src/` onto `sys.path` at import time and imports `match`
  directly (top-level `from match import match`), so `src/` doesn't need
  to become a proper installed package for this to work.
- Same torch-before-faiss import ordering as `build_index.py`/`match.py`,
  for the same OpenMP-segfault reason.
- `GET /health` — trivial liveness check.
- `POST /match` (multipart `file` field):
  1. Rejects anything outside `image/jpeg`, `image/png`, `image/webp` with
     `415`.
  2. Writes the upload to a `tempfile.NamedTemporaryFile` (auto-deleted on
     close) since `match()`/`preprocess_image()` expect a filesystem path,
     not bytes.
  3. Calls `match(path, top_k=5)`; a `ValueError` from `preprocess_image`
     (no face detected, unreadable image) is converted to `422` with the
     original message as `detail`.
  4. Returns `{"matches": [...]}` — same shape `match()` already produces,
     so the response schema needed no separate Pydantic model.
- CORS is restricted to `http://localhost:3000` (the dev frontend origin)
  via `CORSMiddleware`. Would need widening (or an env-driven allowlist)
  before deploying the frontend anywhere else.
- `index.faiss`/`labels.json` are loaded fresh inside `match()` on every
  request — fine at dev/demo QPS. A production server would cache the
  loaded index at startup instead of re-reading it from disk per request.

### `frontend/`
Next.js 16 App Router app (TypeScript, Tailwind, ESLint), scaffolded with
`create-next-app`.

- `src/app/layout.tsx` — root layout, sets the page `<title>`/description.
- `src/app/page.tsx` — server component; renders static heading/copy and
  `<UploadMatcher />`.
- `src/app/upload-matcher.tsx` — the actual feature, as a client
  component (`"use client"`):
  - File `<input>` styled as a dropzone-style label; on change, shows a
    local object-URL preview of the chosen image immediately.
  - POSTs the file to `${NEXT_PUBLIC_API_URL}/match` as `FormData`, tracks
    `idle | loading | error` state.
  - On success, renders the returned matches as a ranked list of cards
    (`#1 Name — 80.3%`), converting `similarity_score` (0-1) to a
    percentage for display.
  - On failure, surfaces the API's `detail` message (e.g. "No face
    detected...") directly rather than a generic error.
- No server-side API route/proxy — the client component calls the FastAPI
  backend directly. Keeping the browser→FastAPI call direct (rather than
  routing it through a Next.js route handler) avoids double-hopping the
  file upload for no benefit, since there's no secret to hide behind a
  proxy here.
- `.env.local` (gitignored) sets `NEXT_PUBLIC_API_URL=http://localhost:8000`.
  The `NEXT_PUBLIC_` prefix is required for a Next.js env var to be
  readable in client-side code at all.

## Known environment gotchas

- This project's Python `.venv` was built with Python 3.12 (via
  Homebrew), not the system Python 3.14. `torch`/`facenet-pytorch`/
  `Pillow` didn't have prebuilt wheels for 3.14 at the time of setup, and
  building from source failed. If dependencies won't install, check the
  Python version first.
- `node`/`npm` weren't installed on this machine at all; Node 26 was
  installed via Homebrew to run `create-next-app` and the dev server.
- Importing `faiss` before `torch` segfaults on macOS (see
  `build_index.py` above) — every entry point that uses both
  (`build_index.py`, `match.py`, `api/main.py`) imports `torch`
  (transitively via `facenet_pytorch`) first.

## Running everything locally

```
# backend (from celebrity-match/)
source .venv/bin/activate
uvicorn api.main:app --port 8000

# frontend (from celebrity-match/frontend/)
npm run dev   # http://localhost:3000
```

## What's intentionally unbuilt

- No fine-tuning (Phase 2) — FaceNet is used purely as a frozen,
  pretrained (`vggface2`) feature extractor.
- No auth, rate limiting, or persistent storage of uploaded selfies —
  uploads live only in a temp file for the duration of one request.
- No deployment config (Docker, hosting, CI) — both apps currently only
  run locally via `uvicorn`/`next dev`.
- No approximate/scalable FAISS index — `IndexFlatIP` is exact and only
  reasonable because the identity count is small (17 in the current
  dataset).
