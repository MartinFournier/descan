#!/usr/bin/env python3
"""
Montage a directory of images into one labelled grid.

Built for eyeballing a whole run at once, e.g. the ``*_detections.png`` overlays
that ``descan-split --debug`` writes.

Example:
    descan-montage ~/Out/debug            # -> ~/Out/debug/montage.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from descan.imageio import read_image, write_png


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Montage a directory of images into one labelled grid."
    )
    parser.add_argument("input_dir", type=Path, help="Directory of images.")
    parser.add_argument(
        "--pattern",
        default="*_detections.png",
        help="Glob for images to include. Default: *_detections.png.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file. Default: <input_dir>/montage.png.",
    )
    parser.add_argument(
        "--columns", type=int, default=7, help="Tiles per row. Default: 7."
    )
    parser.add_argument(
        "--tile-height", type=int, default=300, help="Tile height px. Default: 300."
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(0, 10),
        default=6,
        metavar="0-9",
        help="PNG compression level. Default: 6.",
    )
    return parser.parse_args()


def _tile(path: Path, height: int) -> np.ndarray:
    image = read_image(path)
    scale = height / image.shape[0]
    tile = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(
        tile,
        path.name[:22],
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def build_montage(paths: list[Path], columns: int, tile_height: int) -> np.ndarray:
    """Return a single grid image; tiles are padded to a common width."""
    tiles = [_tile(path, tile_height) for path in paths]
    width = max(tile.shape[1] for tile in tiles)
    fill = (40, 40, 40)

    padded = [
        cv2.copyMakeBorder(
            tile, 0, 0, 0, width - tile.shape[1], cv2.BORDER_CONSTANT, value=fill
        )
        for tile in tiles
    ]
    while len(padded) % columns:
        padded.append(np.full((tile_height, width, 3), fill, np.uint8))

    rows = [np.hstack(padded[i : i + columns]) for i in range(0, len(padded), columns)]
    return np.vstack(rows)


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        logging.error("Not a directory: %s", input_dir)
        return 2

    paths = sorted(input_dir.glob(args.pattern))
    if not paths:
        logging.error("No images matching %s in %s", args.pattern, input_dir)
        return 1

    grid = build_montage(paths, args.columns, args.tile_height)
    output = args.output or input_dir / "montage.png"
    write_png(output, grid, args.png_compression)
    logging.info(
        "Wrote %s (%d tiles, %dx%d)", output, len(paths), grid.shape[1], grid.shape[0]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
