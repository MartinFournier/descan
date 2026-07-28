#!/usr/bin/env python3
"""
Review-and-apply auto-trim of flat white scanner-lid margins on photo crops.

Some crops keep a large white block on one side (the box overshot into the
scanner lid). For each affected crop this shows the original on the left and a
suggested trim on the right; accept or reject with one key.

Only crops the trimmer would actually change are shown, so a folder of ~250 is
a quick pass over the few dozen that need it.

Controls:
    y / ENTER / SPACE / RIGHT   accept the trim (overwrite the file)
    n / LEFT                    keep the original, skip
    f                           keep original and flag for manual cropping
    q / ESC                     quit (remaining files untouched)

Flagged files are moved into a ``to-split/`` subfolder, ready to re-split with
``manualcrop.py <folder>/to-split <folder> --replace``.

Example:
    descan-trim ~/Photos/split
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from descan.ingest import SUPPORTED_EXTENSIONS, find_input_files, read_image, write_png


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review and apply auto-trim of white lid margins on crops."
    )
    parser.add_argument("input_path", type=Path, help="Crop file or directory.")
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories."
    )
    parser.add_argument(
        "--lid-lightness",
        type=float,
        default=225.0,
        metavar="L",
        help="A pixel counts as lid above this Lab lightness. Default: 225.",
    )
    parser.add_argument(
        "--edge-fraction",
        type=float,
        default=0.90,
        metavar="F",
        help="Trim an edge row/col while at least this fraction of it is lid. "
        "Default: 0.90.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.01,
        metavar="FRACTION",
        help="White border kept around the content, relative to the crop's "
        "shorter side. Default: 0.01.",
    )
    parser.add_argument(
        "--display-size",
        type=int,
        default=800,
        metavar="PIXELS",
        help="Height of each panel in the review window. Default: 800.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(0, 10),
        default=6,
        metavar="0-9",
        help="PNG compression level (lossless). Default: 6.",
    )
    return parser.parse_args()


def trim_box(
    image: np.ndarray,
    lid_lightness: float,
    edge_fraction: float,
    margin_fraction: float,
) -> tuple[int, int, int, int]:
    """Return the content box after peeling flat white lid margins off each edge."""
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    chroma = np.abs(lab[:, :, 1].astype(np.float32) - 128.0) + np.abs(
        lab[:, :, 2].astype(np.float32) - 128.0
    )
    lid = (lightness > lid_lightness) & (chroma < 8.0)

    row_lid = lid.mean(axis=1)
    col_lid = lid.mean(axis=0)

    def peel(fractions: np.ndarray) -> tuple[int, int]:
        n = len(fractions)
        low, high = 0, n
        while low < high and fractions[low] > edge_fraction:
            low += 1
        while high > low and fractions[high - 1] > edge_fraction:
            high -= 1
        return low, high

    y0, y1 = peel(row_lid)
    x0, x1 = peel(col_lid)

    if x1 <= x0 or y1 <= y0:  # all lid; leave untouched
        return 0, 0, width, height

    margin = int(round(min(width, height) * margin_fraction))
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width, x1 + margin)
    y1 = min(height, y1 + margin)
    return x0, y0, x1, y1


def panel(image: np.ndarray, height: int, label: str) -> np.ndarray:
    """Scale an image to a fixed height and add a caption bar."""
    scale = height / image.shape[0]
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bar = np.zeros((28, scaled.shape[1], 3), np.uint8)
    cv2.putText(
        bar,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, scaled])


def review(
    queue: list[tuple[Path, tuple[int, int, int, int], np.ndarray]],
    args: argparse.Namespace,
) -> tuple[int, list[Path]]:
    window = "autocrop review"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    accepted = 0
    flagged: list[Path] = []
    index = 0

    while index < len(queue):
        path, (x0, y0, x1, y1), image = queue[index]
        trimmed = image[y0:y1, x0:x1]

        left = panel(image, args.display_size, f"[{index + 1}/{len(queue)}] ORIGINAL")
        right = panel(
            trimmed,
            args.display_size,
            "AUTOCROP  y=accept  n=skip  f=flag  q=quit",
        )
        gap = np.zeros((left.shape[0], 24, 3), np.uint8)
        cv2.imshow(window, np.hstack([left, gap, right]))

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("y"), 13, 32, 83, 84):  # y, enter, space, right/down arrow
            write_png(path, trimmed.copy(), args.png_compression)
            logging.info("trimmed %s -> %dx%d", path.name, x1 - x0, y1 - y0)
            accepted += 1
            index += 1
        elif key in (ord("n"), 81, 82):  # n, left/up arrow
            logging.info("kept %s", path.name)
            index += 1
        elif key == ord("f"):  # flag for manual cropping
            flagged.append(path)
            logging.info("flagged %s", path.name)
            index += 1
        elif key in (ord("q"), 27):  # q, esc
            break

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return accepted, flagged


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_path = args.input_path.expanduser().resolve()
    if input_path.is_dir():
        files = find_input_files(input_path, args.recursive)
    elif input_path.is_file() and input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        files = [input_path]
    else:
        logging.error("No usable input at %s", input_path)
        return 2

    queue = []
    for path in files:
        try:
            image = read_image(path)
        except Exception as error:
            logging.error("%s: %s", path.name, error)
            continue
        height, width = image.shape[:2]
        box = trim_box(image, args.lid_lightness, args.edge_fraction, args.margin)
        x0, y0, x1, y1 = box
        # Only queue crops the trim would meaningfully shrink.
        if (x1 - x0) * (y1 - y0) < 0.97 * width * height:
            queue.append((path, box, image))

    if not queue:
        logging.info("Nothing to trim: no crop had a large white margin.")
        return 0

    logging.info("%d crop(s) to review", len(queue))
    accepted, flagged = review(queue, args)
    logging.info("Applied %d trim(s)", accepted)

    if flagged:
        base = input_path if input_path.is_dir() else input_path.parent
        to_split = base / "to-split"
        to_split.mkdir(exist_ok=True)
        for path in flagged:
            path.rename(to_split / path.name)
        logging.info(
            "Moved %d flagged crop(s) to %s. Re-split them with:\n"
            "  manualcrop.py %s %s --replace",
            len(flagged),
            to_split,
            to_split,
            base,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
