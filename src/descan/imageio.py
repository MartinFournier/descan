"""Image IO: read, write, resize, and path helpers."""

from __future__ import annotations

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
        image = cv2.normalize(  # type: ignore[call-overload]
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
