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
    descan-split ~/Scans ~/Photos/split --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from descan.detect import (
    crop_bounding_box,
    find_photo_detections,
    make_debug_image,
)
from descan.imageio import (
    find_input_files,
    output_path_for,
    read_image,
    write_png,
)
from descan.orient import (
    auto_orient_photo,
    load_face_detector,
    rotate_right_angle,
)


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

    for input_path in tqdm(input_files, unit="scan", disable=not sys.stderr.isatty()):
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
