#!/usr/bin/env python3
"""
Manually split a flatbed scan by drawing rectangles.

For photos the automatic detector in ``ingest.py`` misses or mis-splits. Open a
scan, drag a box around each photo, and each is written as its own PNG using the
same naming and auto-orientation as ``ingest.py``.

Controls (OpenCV's built-in selector):
    drag            draw a box
    ENTER / SPACE   confirm the box, start another
    c               cancel the box being drawn
    ESC             finish this scan and save all boxes

By default new crops are numbered after any existing ``<stem>_pNN.png`` so they
add to, rather than overwrite, the automatic output. Use --overwrite to renumber
from p01.

Example:
    python manualcrop.py ~/Scans/titi_mariage.png ~/Photos/split
    python manualcrop.py ~/Scans ~/Photos/split --recursive
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from ingest import (
    SUPPORTED_EXTENSIONS,
    auto_orient_photo,
    find_input_files,
    load_face_detector,
    read_image,
    rotate_right_angle,
    write_png,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually crop photos out of flatbed scans by drawing boxes."
    )
    parser.add_argument("input_path", type=Path, help="Scan file or directory.")
    parser.add_argument(
        "output_directory", type=Path, help="Directory for the crops."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a directory, recurse into subdirectories.",
    )
    parser.add_argument(
        "--display-size",
        type=int,
        default=1400,
        metavar="PIXELS",
        help="Largest window dimension while drawing. Crops stay full-res. "
        "Default: 1400.",
    )
    parser.add_argument(
        "--no-auto-orient",
        action="store_true",
        help="Disable face-based 90-degree orientation.",
    )
    parser.add_argument(
        "--always-landscape",
        action="store_true",
        help="Rotate portrait crops to landscape when no face decides it.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(0, 10),
        default=6,
        metavar="0-9",
        help="PNG compression level (lossless). Default: 6.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Number crops from p01, replacing existing files, instead of "
        "appending after them.",
    )
    return parser.parse_args()


def next_start_index(output_directory: Path, stem: str) -> int:
    """Return the first free ``_pNN`` index for this scan's crops."""
    highest = 0
    for existing in output_directory.glob(f"{stem}_p*.png"):
        suffix = existing.stem[len(stem) + 2 :]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def select_boxes(
    image: np.ndarray,
    display_size: int,
    window: str,
) -> list[tuple[int, int, int, int]]:
    """Show the scan and return the drawn boxes in full-resolution coordinates."""
    height, width = image.shape[:2]
    scale = min(1.0, display_size / max(height, width))

    if scale < 1.0:
        display = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        display = image.copy()

    # rois: Nx4 array of (x, y, w, h) in display coordinates.
    rois = cv2.selectROIs(window, display, showCrosshair=False, fromCenter=False)

    boxes: list[tuple[int, int, int, int]] = []
    for x, y, w, h in rois:
        if w <= 0 or h <= 0:
            continue
        x0 = int(round(x / scale))
        y0 = int(round(y / scale))
        x1 = min(width, int(round((x + w) / scale)))
        y1 = min(height, int(round((y + h) / scale)))
        boxes.append((max(0, x0), max(0, y0), x1, y1))
    return boxes


def process_scan(path: Path, args: argparse.Namespace, detector) -> int:
    image = read_image(path)

    window = f"{path.name}  -  drag boxes, ENTER after each, ESC when done"
    try:
        boxes = select_boxes(image, args.display_size, window)
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # flush the window close on some backends

    if not boxes:
        logging.info("%s: no boxes drawn; skipped", path.name)
        return 0

    start = 1 if args.overwrite else next_start_index(args.output_directory, path.stem)

    written = 0
    for offset, (x0, y0, x1, y1) in enumerate(boxes):
        index = start + offset
        destination = args.output_directory / f"{path.stem}_p{index:02d}.png"

        if destination.exists() and not args.overwrite:
            logging.warning(
                "%s: exists; skipping (use --overwrite)", destination.name
            )
            continue

        cropped = image[y0:y1, x0:x1].copy()

        rotation = 0
        faces = 0
        if not args.no_auto_orient:
            cropped, rotation, faces = auto_orient_photo(
                cropped, detector, args.always_landscape
            )
        elif args.always_landscape and cropped.shape[0] > cropped.shape[1]:
            cropped = rotate_right_angle(cropped, 1)
            rotation = 90

        write_png(destination, cropped, args.png_compression)
        logging.info(
            "  %s: %dx%d, orientation +%d°, faces=%d",
            destination.name,
            cropped.shape[1],
            cropped.shape[0],
            rotation,
            faces,
        )
        written += 1

    return written


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_path = args.input_path.expanduser().resolve()
    output_directory = args.output_directory.expanduser().resolve()

    if input_path.is_dir():
        scans = find_input_files(input_path, args.recursive)
        if output_directory == input_path:
            logging.error("Input and output directories must differ")
            return 2
    elif input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logging.error("Unsupported file type: %s", input_path.suffix)
            return 2
        scans = [input_path]
    else:
        logging.error("Input path does not exist: %s", input_path)
        return 2

    if not scans:
        logging.error("No supported scans found in %s", input_path)
        return 1

    args.output_directory = output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    detector = None if args.no_auto_orient else load_face_detector()

    total = 0
    for path in scans:
        logging.info("%s: %d already present", path.name, next_start_index(output_directory, path.stem) - 1)
        try:
            total += process_scan(path, args, detector)
        except Exception as error:  # keep going to the next scan
            logging.error("%s: %s", path.name, error)

    logging.info("Wrote %d crop%s", total, "" if total == 1 else "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
