"""Face-based orientation of a cropped photo (YuNet)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np


def load_face_detector() -> cv2.FaceDetectorYN | None:
    """Load the YuNet DNN face detector used for orientation.

    OpenCV 5 removed the legacy Haar ``CascadeClassifier`` from the Python
    build, so orientation uses the bundled YuNet model instead. The ONNX file
    is vendored in the package's ``assets/`` directory.
    """
    model_path = (
        Path(__file__).resolve().parent / "assets" / "face_detection_yunet_2023mar.onnx"
    )

    if not model_path.exists():
        logging.warning(
            "Face model not found at %s; auto-orientation is disabled",
            model_path,
        )
        return None

    if not hasattr(cv2, "FaceDetectorYN"):
        logging.warning("cv2.FaceDetectorYN unavailable; auto-orientation is disabled")
        return None

    try:
        # The input size is reset per image before each detection.
        return cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=500,
        )
    except cv2.error as error:
        logging.warning(
            "Could not load face model (%s); auto-orientation is disabled", error
        )
        return None


def rotate_right_angle(image: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Rotate clockwise in 90-degree increments."""
    quarter_turns %= 4

    if quarter_turns == 0:
        return image
    if quarter_turns == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if quarter_turns == 2:
        return cv2.rotate(image, cv2.ROTATE_180)

    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def face_orientation_score(
    image: np.ndarray,
    detector: cv2.FaceDetectorYN,
) -> tuple[float, int]:
    """
    Score how likely an image is upright based on frontal-face detections.

    YuNet is trained on upright faces, so it detects them far more confidently
    than sideways or inverted ones. The score rewards multiple faces but weights
    large, high-confidence detections most, so a single clear face still wins.
    """
    work = image

    maximum_dimension = max(work.shape[:2])
    if maximum_dimension > 1400:
        scale = 1400.0 / maximum_dimension
        work = cv2.resize(
            work,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    work_height, work_width = work.shape[:2]
    detector.setInputSize((work_width, work_height))
    _, faces = detector.detect(work)

    if faces is None or len(faces) == 0:
        return 0.0, 0

    image_area = float(work_height * work_width)

    # faces columns: x, y, w, h, 5 landmarks (10 values), confidence.
    normalized_area = sum(
        (float(face[2]) * float(face[3])) / image_area for face in faces
    )
    confidence = sum(float(face[-1]) for face in faces)
    score = len(faces) * 1.0 + normalized_area * 25.0 + confidence * 0.5

    return score, len(faces)


def auto_orient_photo(
    image: np.ndarray,
    detector: cv2.FaceDetectorYN | None,
    always_landscape: bool,
) -> tuple[np.ndarray, int, int]:
    """
    Choose among 0, 90, 180, and 270 degrees.

    Returns:
        oriented image, clockwise rotation in degrees, detected face count
    """
    if detector is None:
        if always_landscape and image.shape[0] > image.shape[1]:
            return rotate_right_angle(image, 1), 90, 0
        return image, 0, 0

    results: list[tuple[float, int, int, np.ndarray]] = []

    for quarter_turns in range(4):
        candidate = rotate_right_angle(image, quarter_turns)
        score, face_count = face_orientation_score(candidate, detector)
        results.append((score, face_count, quarter_turns, candidate))

    results.sort(key=lambda item: item[0], reverse=True)
    best_score, best_faces, best_turns, best_image = results[0]
    original_score = next(item[0] for item in results if item[2] == 0)

    # Avoid changing orientation for a weak or ambiguous detection.
    if best_faces == 0:
        if always_landscape and image.shape[0] > image.shape[1]:
            return rotate_right_angle(image, 1), 90, 0
        return image, 0, 0

    second_score = results[1][0]

    clearly_better_than_second = (
        best_score >= second_score * 1.15 or best_score - second_score >= 0.75
    )
    clearly_better_than_original = (
        best_turns == 0
        or best_score >= original_score * 1.15
        or best_score - original_score >= 0.75
    )

    if clearly_better_than_second and clearly_better_than_original:
        return best_image, best_turns * 90, best_faces

    return image, 0, next(item[1] for item in results if item[2] == 0)
