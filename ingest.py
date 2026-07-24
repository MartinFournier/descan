#!/usr/bin/env python3
"""
Split flatbed scans containing multiple printed photos.

Features:
- Input files are read-only.
- Detects multiple photos per scan.
- Deskews photos rotated by a few degrees.
- Applies a perspective crop.
- Optionally chooses 0/90/180/270-degree orientation using face detection.
- Writes lossless PNG output.
- Names output from the scan filename plus "_pNN".
- Can produce annotated debug images.

Example:
    python split_scanned_photos.py ~/Scans ~/Photos/split --debug

Dependencies:
    python -m pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
        description="Detect, deskew, crop, and split photos from flatbed scans."
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
        help=(
            "Minimum photo area as a fraction of the complete scan. "
            "Default: 0.012."
        ),
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
        "--background-threshold",
        type=float,
        default=20.0,
        metavar="VALUE",
        help=(
            "Lab-colour distance used to separate photos from the scanner "
            "background. Lower values are more sensitive. Default: 20."
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
        image = np.clip(bgr * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(
            np.uint8
        )
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


def build_detection_mask(
    image: np.ndarray,
    background_threshold: float,
) -> np.ndarray:
    """
    Build a mask whose connected components are individual printed photos.

    Family-album scans are typically several white-bordered prints on a
    near-white scanner lid, so a background-colour difference is close to zero
    and useless. Instead this combines faint print-border edges with a
    saturation/darkness signal from the photo interior to mark foreground, then
    severs the thin bridges (drop shadows, scanner banding) that would
    otherwise weld neighbouring prints into a single blob. Each surviving
    component is one photo.

    The returned mask is 8-bit, 255 on photo regions and 0 on background.
    """
    height, width = image.shape[:2]
    short_side = min(height, width)

    # Denoise first: this removes low-amplitude scanner banding that would
    # otherwise register as spurious texture, while keeping real photo edges.
    denoised = cv2.bilateralFilter(image, 9, 50, 50)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Faint print-border edges. Low thresholds are safe because banding is
    # already denoised away.
    edges = cv2.Canny(gray, 12, 40)

    # Photo interiors depart from the white lid in colour (saturation) or
    # brightness (darkness); its outline reinforces the border edges above.
    saturation = np.abs(lab[:, :, 1] - 128.0) + np.abs(lab[:, :, 2] - 128.0)
    darkness = 255.0 - lab[:, :, 0]
    interior = (
        (saturation > 18.0) | (darkness > 35.0)
    ).astype(np.uint8) * 255
    interior_outline = cv2.morphologyEx(
        interior,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    combined = cv2.bitwise_or(edges, interior_outline)

    # Thicken and close the outline into filled photo regions.
    dilate_size = max(3, int(short_side * 0.006)) | 1
    combined = cv2.dilate(
        combined,
        cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size)),
        iterations=1,
    )
    close_size = (dilate_size * 2) | 1
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size)),
        iterations=1,
    )

    # Open to sever the thin bridges linking adjacent prints so that each photo
    # becomes its own connected component. The kernel is wider than a drop
    # shadow or gap fill but far narrower than a photo.
    open_size = max(5, int(height * 0.014)) | 1
    core = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
        iterations=1,
    )

    return core


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """
    Order four points as top-left, top-right, bottom-right, bottom-left.
    """
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[3] = points[np.argmax(coordinate_difference)]

    return ordered


def quad_area(points: np.ndarray) -> float:
    return abs(float(cv2.contourArea(points.astype(np.float32))))


def find_photo_detections(
    image: np.ndarray,
    minimum_area_fraction: float,
    maximum_photos: int,
    background_threshold: float,
    processing_size: int,
) -> tuple[list[Detection], np.ndarray, float]:
    """Detect candidate printed-photo rectangles."""
    working, scale = resize_for_detection(image, processing_size)
    working_height, working_width = working.shape[:2]
    image_area = float(working_width * working_height)

    mask = build_detection_mask(working, background_threshold)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    candidates: list[Detection] = []

    for label in range(1, component_count):
        component_area = float(stats[label, cv2.CC_STAT_AREA])

        # A quick pre-filter before the more expensive rectangle fit.
        if component_area / image_area < minimum_area_fraction * 0.5:
            continue

        rows, cols = np.where(labels == label)
        points = np.column_stack([cols, rows]).astype(np.float32)
        rectangle = cv2.minAreaRect(points)
        (_, _), (rect_w, rect_h), _ = rectangle

        if rect_w < 8 or rect_h < 8:
            continue

        rect_area = rect_w * rect_h
        fill = component_area / rect_area
        aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
        area_fraction = rect_area / image_area

        # A photo is a well-filled, plausibly-proportioned rectangle. This also
        # rejects the near-full-page blob that appears when detection fails.
        if not (
            minimum_area_fraction < area_fraction < 0.94
            and fill > 0.45
            and aspect_ratio < 6.5
        ):
            continue

        quad = order_quad_points(cv2.boxPoints(rectangle))
        full_resolution_quad = quad / scale
        centre = full_resolution_quad.mean(axis=0)
        area = quad_area(full_resolution_quad)

        candidates.append(
            Detection(
                points=full_resolution_quad,
                area=area,
                centre_x=float(centre[0]),
                centre_y=float(centre[1]),
            )
        )

    # Merge overlapping rectangles, keeping the larger. A single photo whose
    # interior split into two cores (for example a plain sky above a subject)
    # produces overlapping rectangles that collapse back into one here.
    candidates.sort(key=lambda item: item.area, reverse=True)
    unique: list[Detection] = []

    for candidate in candidates:
        overlaps = False
        for accepted in unique:
            intersection, _ = cv2.intersectConvexConvex(
                candidate.points.astype(np.float32),
                accepted.points.astype(np.float32),
            )
            smaller = min(candidate.area, accepted.area)
            if smaller > 0 and intersection > 0.3 * smaller:
                overlaps = True
                break
        if not overlaps:
            unique.append(candidate)

    unique = unique[:maximum_photos]

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


def expanded_quad(
    points: np.ndarray,
    margin_pixels: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Expand a quadrilateral outwards from its centre."""
    points = order_quad_points(points)
    centre = points.mean(axis=0)

    expanded = points.copy()

    for index, point in enumerate(points):
        direction = point - centre
        distance = np.linalg.norm(direction)

        if distance > 0:
            expanded[index] = point + direction / distance * margin_pixels

    expanded[:, 0] = np.clip(expanded[:, 0], 0, image_width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, image_height - 1)

    return expanded.astype(np.float32)


def warp_photo(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Deskew and perspective-correct a detected photo."""
    points = order_quad_points(points)
    top_left, top_right, bottom_right, bottom_left = points

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)

    output_width = max(1, int(round(max(width_top, width_bottom))))
    output_height = max(1, int(round(max(height_left, height_right))))

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    transformation = cv2.getPerspectiveTransform(points, destination)

    return cv2.warpPerspective(
        image,
        transformation,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def load_face_detector() -> "cv2.FaceDetectorYN | None":
    """Load the YuNet DNN face detector used for orientation.

    OpenCV 5 removed the legacy Haar ``CascadeClassifier`` from the Python
    build, so orientation uses the bundled YuNet model instead. The ONNX file
    is vendored alongside this script in ``assets/``.
    """
    model_path = (
        Path(__file__).resolve().parent
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
        logging.warning(
            "cv2.FaceDetectorYN unavailable; auto-orientation is disabled"
        )
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
        logging.warning("Could not load face model (%s); auto-orientation is disabled", error)
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
    detector: "cv2.FaceDetectorYN",
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
    detector: "cv2.FaceDetectorYN | None",
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
        best_score >= second_score * 1.15
        or best_score - second_score >= 0.75
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
    face_detector: "cv2.FaceDetectorYN | None",
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
        background_threshold=args.background_threshold,
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

        points = expanded_quad(
            detection.points,
            margin_pixels,
            width,
            height,
        )
        cropped = warp_photo(image, points)

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
