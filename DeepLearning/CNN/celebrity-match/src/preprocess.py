"""Face detection, alignment, and cropping for the celebrity match pipeline."""

from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps, UnidentifiedImageError

# Loaded once at import time and reused across calls — MTCNN weights don't
# need to be reloaded per image.
#   image_size=160     final crop size FaceNet expects.
#   select_largest=True if multiple faces are found, keep only the biggest
#                       (closest/most prominent) one instead of erroring.
#   post_process=True  apply MTCNN's fixed image standardization, so the
#                       tensor comes out already normalized to ~[-1, 1].
_mtcnn = MTCNN(image_size=160, margin=0, select_largest=True, post_process=True)


def preprocess_image(image_path: Path) -> torch.Tensor:
    """Detect, align, and crop the largest face in an image to a 160x160 tensor.

    Args:
        image_path: Path to the input image.

    Returns:
        A (3, 160, 160) float tensor, normalized and ready for FaceNet.

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If the image cannot be opened or no face is detected.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path)
    except UnidentifiedImageError as exc:
        # File exists but isn't a valid/openable image (corrupt, wrong
        # extension, not actually an image, etc).
        raise ValueError(f"Could not read image: {image_path}") from exc

    # Phone photos store rotation as EXIF metadata rather than rotating the
    # pixels; without this, a portrait selfie can load sideways and MTCNN
    # won't find a face in it.
    img = ImageOps.exif_transpose(img).convert("RGB")

    # MTCNN returns None (rather than raising) when it can't find a face,
    # so we convert that into an explicit error the caller can catch.
    face = _mtcnn(img)
    if face is None:
        raise ValueError(f"No face detected in image: {image_path}")

    return face
