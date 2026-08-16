# Celebrity Face Match — Project Brief for Claude Code

## Project Goal
Build a web app where a user uploads a selfie and gets matched to the
most similar celebrity face using FaceNet embeddings + FAISS similarity search.

# scaffold the project structure first, then build preprocess.py" — do it file by file, not all at once
## Current Phase
Phase 1 — Local pipeline validation (no training, no GPU required)

## Tech Stack
- Python 3.9+
- facenet-pytorch (InceptionResnetV1, pretrained on vggface2)
- MTCNN (face detection + alignment, from facenet-pytorch)
- faiss-cpu (similarity search)
- PyTorch CPU only
- Pillow, numpy, opencv-python

## Dataset
Hollywood Celebrity Face Recognition Dataset from Kaggle
(bhaveshmittal/celebrity-face-recognition-dataset)
Already downloaded and unzipped to: data/celebrities/
Structure: one folder per celebrity, images inside

## Project Structure to Scaffold
celebrity-match/
├── data/
│   └── celebrities/          # already populated
├── embeddings/
│   ├── index.faiss           # to be generated
│   └── labels.json           # to be generated
├── src/
│   ├── preprocess.py         # MTCNN face detection, align, crop to 160x160
│   ├── build_index.py        # embed all celeb images → average per identity → save FAISS index + labels
│   └── match.py              # load index, embed query image, return top-5 matches with scores
├── tests/
│   └── test_selfies/         # manually drop test images here
├── requirements.txt
└── README.md

## Scripts to Build

### preprocess.py
- Input: image path
- Detect face using MTCNN
- Align and crop to 160x160
- Return normalized tensor ready for FaceNet
- Handle: no face detected, multiple faces (pick largest), bad image

### build_index.py
- Walk data/celebrities/ folder
- For each celebrity identity (folder = label):
  - Run preprocess.py on each image
  - Pass through FaceNet → 512-dim embedding
  - Average embeddings across all images for that identity
- Store averaged embeddings in FAISS IndexFlatIP (inner product = cosine on normalized vectors)
- Save index to embeddings/index.faiss
- Save label list to embeddings/labels.json
- Print summary: N celebrities indexed, M images processed, K failed/skipped

### match.py
- Input: path to a query image (selfie)
- Run preprocess.py on it
- Embed with FaceNet
- Search FAISS index for top 5 nearest neighbors
- Return: list of {celebrity_name, similarity_score} dicts
- Print results to console in a readable format

## Validation Goal
Run match.py on 3-5 test selfies of known people and confirm:
- Top match is plausible
- Similarity scores are in a reasonable range (0.5-1.0 for good matches)
- Pipeline runs end-to-end without errors

## What Comes After Phase 1 (context only, don't build yet)
- Phase 2: Fine-tune FaceNet on VGGFace2 dataset (on Kaggle GPU)
- Phase 3: FastAPI backend wrapping match.py
- Phase 4: Next.js frontend with upload UI and match result cards

## Notes
- Keep all code modular — preprocess.py and match.py will be reused in Phase 2 and 3 unchanged
- Add docstrings to every function
- Use pathlib throughout, not os.path
- Print progress during build_index.py (tqdm preferred)
- Normalize all embeddings to unit length before storing in FAISS

## Additional Instruction 
Scaffold the project structure first, then build preprocess.py" — do it file by file, not all at once

