#!/usr/bin/env python3
"""
Split flatbed scans containing multiple printed photos.

Features:
- Input files are read-only.
- Detects multiple photos per scan and splits them into separate files.
- Photos are kept whole and uncropped: each output is an axis-aligned slice of
  the source, not a tight or deskewed crop.
- Optionally chooses 0/90/180/270-degree orientation using face detection.
- Writes lossless PNG output.
- Names output from the scan filename plus "_pNN".
- Can produce annotated debug images.

Example:
    python ingest.py ~/Scans ~/Photos/split --debug

Dependencies:
    python -m pip install "opencv-python>=5" numpy
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


@dataclass
class Detection:
    """Detected photo represented in full-resolution source coordinates."""

    points: np.ndarray
    area: float
    centre_x: float
    centre_y: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the individual photos out of flatbed scans."
    )

    parser.add_argument(
        "input_directory",
        type=Path,
        help="Directory containing the original scan files.",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="Directory in which split PNG files will be written.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process subdirectories recursively.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.012,
        metavar="FRACTION",
        help=("Minimum photo area as a fraction of the complete scan. Default: 0.012."),
    )
    parser.add_argument(
        "--max-photos",
        type=int,
        default=8,
        help="Maximum detections accepted from one scan. Default: 8.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.003,
        metavar="FRACTION",
        help=(
            "Extra margin retained around each detected photo, relative to "
            "the scan's shorter dimension. Default: 0.003."
        ),
    )
    parser.add_argument(
        "--processing-size",
        type=int,
        default=1200,
        metavar="PIXELS",
        help=(
            "Maximum dimension used during detection. Full resolution is "
            "still used for output. Default: 1200."
        ),
    )
    parser.add_argument(
        "--no-auto-orient",
        action="store_true",
        help="Disable face-based 90-degree orientation.",
    )
    parser.add_argument(
        "--always-landscape",
        action="store_true",
        help=(
            "When no face is detected, rotate portrait crops to landscape. "
            "Normally their detected orientation is preserved."
        ),
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(0, 10),
        default=6,
        metavar="0-9",
        help=(
            "PNG compression level. All levels are lossless; higher values "
            "take longer and generally produce smaller files. Default: 6."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write annotated detection images into an output/debug directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output PNG files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect photos and report filenames without writing photo crops.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable additional diagnostic logging.",
    )

    args = parser.parse_args()

    if not 0.001 <= args.min_area <= 0.5:
        parser.error("--min-area must be between 0.001 and 0.5")

    if args.max_photos < 1:
        parser.error("--max-photos must be at least 1")

    if not 0.0 <= args.margin <= 0.05:
        parser.error("--margin must be between 0 and 0.05")

    if args.processing_size < 500:
        parser.error("--processing-size must be at least 500")

    return args


def read_image(path: Path) -> np.ndarray:
    """
    Read an image without relying on OpenCV's filename handling.

    np.fromfile also works with paths containing non-ASCII characters on
    platforms where cv2.imread may have trouble with them.
    """
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("OpenCV could not decode the file")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # Composite transparency against white.
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        image = np.clip(bgr * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    elif image.shape[2] != 3:
        raise ValueError(f"unsupported channel count: {image.shape[2]}")

    # Normalize 16-bit scans to 8-bit because the detection and face model
    # work on 8-bit images. The PNG result is therefore also 8-bit.
    if image.dtype == np.uint16:
        image = (image / 257).round().astype(np.uint8)
    elif image.dtype != np.uint8:
        image = cv2.normalize(
            image,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint8)

    return image


def write_png(path: Path, image: np.ndarray, compression: int) -> None:
    """Write an image as PNG, supporting non-ASCII paths."""
    success, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, compression],
    )

    if not success:
        raise OSError("OpenCV failed to encode PNG data")

    encoded.tofile(path)


def resize_for_detection(
    image: np.ndarray,
    maximum_dimension: int,
) -> tuple[np.ndarray, float]:
    """
    Return a downscaled working copy and its scale relative to the source.

    A scale of 0.5 means one working-image pixel represents two source pixels.
    """
    height, width = image.shape[:2]
    largest = max(height, width)

    if largest <= maximum_dimension:
        return image.copy(), 1.0

    scale = maximum_dimension / largest
    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill enclosed holes in a binary mask, solidifying each region.

    Flood the background inward from a guaranteed-background border (a padded
    frame, since a photo may touch the real image corner), then whatever the
    flood could not reach is an interior hole and gets turned on. This is what
    makes each photo a single solid blob even when its interior contains light
    sky or white clothing.
    """
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, np.zeros((0, 0), np.uint8), (0, 0), 255)
    background = flooded[1:-1, 1:-1]
    return mask | cv2.bitwise_not(background)


def build_detection_mask(image: np.ndarray) -> np.ndarray:
    """
    Build a mask whose connected components are individual printed photos.

    Family-album scans are white-bordered prints on a near-white scanner lid.
    The approach is deliberately simple: mark every pixel that is *not*
    background (darker than the lid, or coloured), fill each photo solid so its
    light interior regions are included, then open away thin links and speckle.
    Each surviving component is one whole photo.

    The returned mask is 8-bit, 255 on photo regions and 0 on background.
    """
    height, width = image.shape[:2]
    short_side = min(height, width)

    # Denoise to suppress low-amplitude scanner banding.
    denoised = cv2.bilateralFilter(image, 9, 50, 50)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[:, :, 0]
    chroma = np.abs(lab[:, :, 1] - 128.0) + np.abs(lab[:, :, 2] - 128.0)

    # Background lightness estimated from the scan border, which is almost
    # always lid rather than photo.
    border = np.concatenate(
        [
            lightness[:3].ravel(),
            lightness[-3:].ravel(),
            lightness[:, :3].ravel(),
            lightness[:, -3:].ravel(),
        ]
    )
    background_lightness = float(np.median(border))

    # Foreground: noticeably darker than the lid, or noticeably coloured.
    foreground = (((background_lightness - lightness) > 12.0) | (chroma > 10.0)).astype(
        np.uint8
    ) * 255

    # Close small speckle gaps, fill each photo solid, then open to drop thin
    # bridges between adjacent prints and isolated noise.
    close_size = max(3, int(short_side * 0.010)) | 1
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    foreground = fill_holes(foreground)

    open_size = max(3, int(short_side * 0.012)) | 1
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )

    return foreground


def quad_area(points: np.ndarray) -> float:
    return abs(float(cv2.contourArea(points.astype(np.float32))))


def find_photo_detections(
    image: np.ndarray,
    minimum_area_fraction: float,
    maximum_photos: int,
    processing_size: int,
) -> tuple[list[Detection], np.ndarray, float]:
    """Detect candidate printed-photo rectangles."""
    working, scale = resize_for_detection(image, processing_size)
    working_height, working_width = working.shape[:2]
    image_area = float(working_width * working_height)

    mask = build_detection_mask(working)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    # Axis-aligned bounding boxes, in working-image coordinates. We deliberately
    # avoid minAreaRect: fitting a rotated rectangle to an irregular or partial
    # component produces skewed slivers over sub-regions of a single photo. The
    # goal here is only to split the scan into whole, uncropped photos, so an
    # upright box that keeps the full print (border included) is what we want.
    boxes: list[tuple[float, float, float, float]] = []

    for label in range(1, component_count):
        component_area = float(stats[label, cv2.CC_STAT_AREA])
        if component_area / image_area < minimum_area_fraction * 0.5:
            continue

        x = float(stats[label, cv2.CC_STAT_LEFT])
        y = float(stats[label, cv2.CC_STAT_TOP])
        w = float(stats[label, cv2.CC_STAT_WIDTH])
        h = float(stats[label, cv2.CC_STAT_HEIGHT])

        if w < 8 or h < 8:
            continue

        area_fraction = (w * h) / image_area
        aspect_ratio = max(w, h) / min(w, h)

        # A photo is a plausibly-proportioned box. The upper area bound also
        # rejects the near-full-page blob that appears when detection fails.
        if not (minimum_area_fraction < area_fraction < 0.94 and aspect_ratio < 6.5):
            continue

        boxes.append((x, y, x + w, y + h))

    # Merge overlapping boxes into one per photo. This collapses the skewed
    # slivers and sky/subject splits that previously became separate detections.
    boxes = merge_overlapping_boxes(boxes, overlap_fraction=0.12)

    # The mask hugs the coloured/dark content, so the box clips a photo's white
    # border and pale regions (sky, white clothing). Grow each box outward until
    # it reaches the flat white lid or a neighbouring photo, recovering the whole
    # print without merging separate ones.
    boxes = expand_boxes_to_lid(working, boxes)

    # A photo with a low-content band running through it (plain sand, a pale
    # wall) splits into adjacent boxes. Rejoin boxes whose shared border is photo
    # content rather than white lid; separate prints have a lid gap between them.
    boxes = merge_content_bridged_boxes(working, boxes)

    candidates: list[Detection] = []
    for x0, y0, x1, y1 in boxes:
        points = (
            np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.float32,
            )
            / scale
        )
        centre = points.mean(axis=0)
        candidates.append(
            Detection(
                points=points,
                area=quad_area(points),
                centre_x=float(centre[0]),
                centre_y=float(centre[1]),
            )
        )

    candidates.sort(key=lambda item: item.area, reverse=True)
    unique = candidates[:maximum_photos]

    # Reading order: cluster approximately into rows, then left to right.
    if unique:
        typical_height = np.median(
            [
                max(
                    np.linalg.norm(item.points[3] - item.points[0]),
                    np.linalg.norm(item.points[2] - item.points[1]),
                )
                for item in unique
            ]
        )
        row_tolerance = max(30.0, float(typical_height) * 0.35)

        unique.sort(
            key=lambda item: (
                round(item.centre_y / row_tolerance),
                item.centre_x,
            )
        )

    return unique, mask, scale


def expand_boxes_to_lid(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """
    Grow each detection box outward until it meets the scanner lid or a neighbour.

    The detection mask keys on coloured/dark content, so a box hugs that content
    and clips the print's white border and pale regions (sky, white clothing).
    Each side is pushed out in small steps and stops when the strip just beyond
    is mostly flat white lid, when it reaches another box, or at the image edge.
    Because it stops at the white gap, separate prints are never merged.
    """
    if not boxes:
        return boxes

    height, width = image.shape[:2]
    lab = cv2.cvtColor(cv2.bilateralFilter(image, 9, 50, 50), cv2.COLOR_BGR2LAB).astype(
        np.float32
    )
    lightness = lab[:, :, 0]
    chroma = np.abs(lab[:, :, 1] - 128.0) + np.abs(lab[:, :, 2] - 128.0)

    # Lid reference: the scan is mostly lid, so a high lightness percentile is
    # a robust white estimate. "Lid" is bright and near-neutral.
    lid_reference = float(np.percentile(lightness, 92))
    is_lid = ((lightness > lid_reference - 12.0) & (chroma < 7.0)).astype(np.uint8)

    step = max(2, int(min(height, width) * 0.004))
    lid_strip_fraction = 0.80

    def blocked(strip: np.ndarray) -> bool:
        return strip.size > 0 and float(strip.mean()) > lid_strip_fraction

    expanded: list[tuple[float, float, float, float]] = []
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        others = [o for j, o in enumerate(boxes) if j != index]

        def hits_neighbour(nx0, ny0, nx1, ny1, others=others) -> bool:
            for ox0, oy0, ox1, oy1 in others:
                if not (ox1 <= nx0 or ox0 >= nx1 or oy1 <= ny0 or oy0 >= ny1):
                    return True
            return False

        while y0 - step >= 0 and not blocked(is_lid[y0 - step : y0, x0:x1]):
            if hits_neighbour(x0, y0 - step, x1, y0):
                break
            y0 -= step
        while y1 + step <= height and not blocked(is_lid[y1 : y1 + step, x0:x1]):
            if hits_neighbour(x0, y1, x1, y1 + step):
                break
            y1 += step
        while x0 - step >= 0 and not blocked(is_lid[y0:y1, x0 - step : x0]):
            if hits_neighbour(x0 - step, y0, x0, y1):
                break
            x0 -= step
        while x1 + step <= width and not blocked(is_lid[y0:y1, x1 : x1 + step]):
            if hits_neighbour(x1, y0, x1 + step, y1):
                break
            x1 += step

        expanded.append((float(x0), float(y0), float(x1), float(y1)))

    return expanded


def merge_content_bridged_boxes(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """
    Rejoin boxes that are two parts of one photo split by a low-content band.

    When a plain sand strip or a dark wall runs through a photo, the mask breaks
    it into adjacent boxes. For an adjacent pair, if the *whole union rectangle*
    is almost entirely photo content (very little white lid inside it) the two
    boxes are one photo and are merged. Two separate prints leave a white lid
    gap inside the union, so it is not merged.

    The union test is deliberately strict (a low lid fraction). It fires on
    clearly dark over-splits but abstains when a photo has large bright regions
    that could be confused with a gap, and it protects pale/faded prints — which
    read as lid-like — from ever being welded together (losing a photo is far
    worse than an over-split, which is just an extra file). Iterates to a fixed
    point.
    """
    if len(boxes) < 2:
        return boxes

    height, width = image.shape[:2]
    lab = cv2.cvtColor(cv2.bilateralFilter(image, 9, 50, 50), cv2.COLOR_BGR2LAB).astype(
        np.float32
    )
    lightness = lab[:, :, 0]
    chroma = np.abs(lab[:, :, 1] - 128.0) + np.abs(lab[:, :, 2] - 128.0)
    lid_reference = float(np.percentile(lightness, 92))
    is_lid = ((lightness > lid_reference - 12.0) & (chroma < 7.0)).astype(np.uint8)

    boxes = [[int(round(v)) for v in box] for box in boxes]

    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                y_overlap = min(a[3], b[3]) - max(a[1], b[1])
                x_overlap = min(a[2], b[2]) - max(a[0], b[0])
                min_height = min(a[3] - a[1], b[3] - b[1])
                min_width = min(a[2] - a[0], b[2] - b[0])

                side_by_side = (
                    y_overlap > 0.3 * min_height
                    and (max(a[0], b[0]) - min(a[2], b[2])) < 0.15 * min_width
                )
                stacked = (
                    x_overlap > 0.3 * min_width
                    and (max(a[1], b[1]) - min(a[3], b[3])) < 0.15 * min_height
                )

                if not (side_by_side or stacked):
                    continue

                union = is_lid[
                    min(a[1], b[1]) : max(a[3], b[3]),
                    min(a[0], b[0]) : max(a[2], b[2]),
                ]
                if union.size and float(union.mean()) < 0.10:
                    boxes[i] = [
                        min(a[0], b[0]),
                        min(a[1], b[1]),
                        max(a[2], b[2]),
                        max(a[3], b[3]),
                    ]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break

    return [tuple(float(v) for v in box) for box in boxes]


def merge_overlapping_boxes(
    boxes: list[tuple[float, float, float, float]],
    overlap_fraction: float,
) -> list[tuple[float, float, float, float]]:
    """
    Union axis-aligned boxes that overlap by more than ``overlap_fraction`` of
    the smaller box, iterating to a fixed point.

    Fragments of one photo (a sliver detection, or a sky region split from its
    subject) overlap heavily and collapse into a single box, while genuinely
    separate prints — kept apart by the white gap between them — do not.
    """
    boxes = list(boxes)
    changed = True

    while changed:
        changed = False
        result: list[tuple[float, float, float, float]] = []

        for box in boxes:
            for index, other in enumerate(result):
                ix0 = max(box[0], other[0])
                iy0 = max(box[1], other[1])
                ix1 = min(box[2], other[2])
                iy1 = min(box[3], other[3])

                intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                area_box = (box[2] - box[0]) * (box[3] - box[1])
                area_other = (other[2] - other[0]) * (other[3] - other[1])
                smaller = min(area_box, area_other)

                if smaller > 0 and intersection > overlap_fraction * smaller:
                    result[index] = (
                        min(box[0], other[0]),
                        min(box[1], other[1]),
                        max(box[2], other[2]),
                        max(box[3], other[3]),
                    )
                    changed = True
                    break
            else:
                result.append(box)

        boxes = result

    return boxes


def crop_bounding_box(
    image: np.ndarray,
    points: np.ndarray,
    margin_pixels: float,
) -> np.ndarray:
    """
    Return the uncropped photo as an axis-aligned slice of the source image.

    The detected box is padded by ``margin_pixels`` and clipped to the image, so
    the whole print is kept. No deskew or perspective correction is applied;
    splitting the scan is the only job here.
    """
    height, width = image.shape[:2]

    x0 = int(np.floor(points[:, 0].min() - margin_pixels))
    y0 = int(np.floor(points[:, 1].min() - margin_pixels))
    x1 = int(np.ceil(points[:, 0].max() + margin_pixels))
    y1 = int(np.ceil(points[:, 1].max() + margin_pixels))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x1)
    y1 = min(height, y1)

    return image[y0:y1, x0:x1].copy()


def load_face_detector() -> cv2.FaceDetectorYN | None:
    """Load the YuNet DNN face detector used for orientation.

    OpenCV 5 removed the legacy Haar ``CascadeClassifier`` from the Python
    build, so orientation uses the bundled YuNet model instead. The ONNX file
    is vendored in the repo's ``assets/`` directory.
    """
    model_path = (
        Path(__file__).resolve().parent.parent
        / "assets"
        / "face_detection_yunet_2023mar.onnx"
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


def make_debug_image(
    image: np.ndarray,
    detections: Sequence[Detection],
    maximum_dimension: int = 2200,
) -> np.ndarray:
    """Draw detected photo boundaries and their output ordering."""
    debug, scale = resize_for_detection(image, maximum_dimension)

    for index, detection in enumerate(detections, start=1):
        points = np.round(detection.points * scale).astype(np.int32)
        cv2.polylines(
            debug,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=max(2, int(max(debug.shape[:2]) / 900)),
        )

        label_point = tuple(points[0])
        cv2.putText(
            debug,
            f"p{index:02d}",
            (label_point[0] + 8, label_point[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return debug


def output_path_for(
    output_directory: Path,
    input_path: Path,
    index: int,
) -> Path:
    return output_directory / f"{input_path.stem}_p{index:02d}.png"


def find_input_files(directory: Path, recursive: bool) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()

    return sorted(
        (
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )


def process_scan(
    input_path: Path,
    output_directory: Path,
    args: argparse.Namespace,
    face_detector: cv2.FaceDetectorYN | None,
) -> tuple[int, int]:
    """
    Process one scanner image.

    Returns:
        number detected, number written
    """
    image = read_image(input_path)
    height, width = image.shape[:2]

    detections, _, _ = find_photo_detections(
        image=image,
        minimum_area_fraction=args.min_area,
        maximum_photos=args.max_photos,
        processing_size=args.processing_size,
    )

    if not detections:
        logging.warning("%s: no photos detected", input_path.name)
        return 0, 0

    logging.info(
        "%s: detected %d photo%s",
        input_path.name,
        len(detections),
        "" if len(detections) == 1 else "s",
    )

    if args.debug:
        debug_directory = output_directory / "debug"
        debug_directory.mkdir(parents=True, exist_ok=True)

        debug_image = make_debug_image(image, detections)
        debug_path = debug_directory / f"{input_path.stem}_detections.png"
        write_png(debug_path, debug_image, args.png_compression)

    margin_pixels = args.margin * min(width, height)
    written = 0

    for index, detection in enumerate(detections, start=1):
        destination = output_path_for(output_directory, input_path, index)

        if destination.exists() and not args.overwrite:
            logging.warning(
                "%s: output exists; skipping. Use --overwrite to replace it",
                destination.name,
            )
            continue

        cropped = crop_bounding_box(image, detection.points, margin_pixels)

        rotation = 0
        face_count = 0

        if not args.no_auto_orient:
            cropped, rotation, face_count = auto_orient_photo(
                cropped,
                face_detector,
                args.always_landscape,
            )
        elif args.always_landscape and cropped.shape[0] > cropped.shape[1]:
            cropped = rotate_right_angle(cropped, 1)
            rotation = 90

        logging.info(
            "  %s: %dx%d, orientation +%d°, faces=%d",
            destination.name,
            cropped.shape[1],
            cropped.shape[0],
            rotation,
            face_count,
        )

        if not args.dry_run:
            write_png(destination, cropped, args.png_compression)

        written += 1

    return len(detections), written


def main() -> int:
    args = parse_arguments()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_directory = args.input_directory.expanduser().resolve()
    output_directory = args.output_directory.expanduser().resolve()

    if not input_directory.is_dir():
        logging.error("Input directory does not exist: %s", input_directory)
        return 2

    if input_directory == output_directory:
        logging.error("Input and output directories must be different")
        return 2

    output_directory.mkdir(parents=True, exist_ok=True)

    input_files = find_input_files(input_directory, args.recursive)

    if not input_files:
        logging.error("No supported image files found in %s", input_directory)
        return 1

    face_detector = None
    if not args.no_auto_orient:
        face_detector = load_face_detector()

    scans_processed = 0
    photos_detected = 0
    photos_written = 0
    failures = 0

    for input_path in input_files:
        try:
            detected, written = process_scan(
                input_path,
                output_directory,
                args,
                face_detector,
            )
            scans_processed += 1
            photos_detected += detected
            photos_written += written
        except Exception as error:
            failures += 1
            logging.error("%s: %s", input_path.name, error)
            if args.verbose:
                logging.exception("Detailed failure")

    action = "would write" if args.dry_run else "wrote"

    logging.info(
        "Finished: %d scans, %d photos detected, %s %d files, %d failures",
        scans_processed,
        photos_detected,
        action,
        photos_written,
        failures,
    )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
