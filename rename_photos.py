#!/usr/bin/env python3
"""
Rename photos to ``YYYY-MM-DD_<Name>_pNNN__<cleaned-original>.ext``.

The date is read from each photo's darktable ``.xmp`` sidecar (or the image's
own EXIF if there is no sidecar), via exiftool. Files are numbered ``pNNN`` in
ascending capture-date order. Each image's sidecar is renamed alongside it so
darktable keeps the edit history linked.

The original name is cleaned: a leading ``titi_`` is dropped, as are the crop
suffixes (``_pNN``, including repeated ones from re-splits), the ``2x/3x/4x/6x``
sheet markers, and trailing split indices (``_1``, ``_2`` ...).

    titi_3x_famille_p02.png (+ .xmp)   -> 1994-08-28_Denise_p007__famille.png (+ .xmp)
    titi_voilier_p03_p01.png           -> ..._Denise_pNNN__voilier.png

Dry-run by default; pass --apply to actually rename.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

from ingest import find_input_files

# EXIF/XMP tags tried in order for a capture date.
DATE_TAGS = ["DateTimeOriginal", "CreateDate", "DateCreated", "MediaCreateDate"]
PLACEHOLDER_DATE = "0000-00-00"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename photos (and darktable .xmp sidecars) with a date prefix."
    )
    parser.add_argument("input_path", type=Path, help="Folder of photos.")
    parser.add_argument("--name", default="Denise", help="Subject name. Default: Denise.")
    parser.add_argument(
        "--fallback-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Date for files with no capture date. "
        f"Without it, undated files get {PLACEHOLDER_DATE} and sort first.",
    )
    parser.add_argument(
        "--start", type=int, default=1, help="First pNNN index. Default: 1."
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename. Without it, only prints the planned renames.",
    )
    return parser.parse_args()


def clean_original(stem: str) -> str:
    """Strip titi_, crop suffixes, sheet markers, and split indices."""
    stem = re.sub(r"(?i)_p\d+", "", stem)           # _pNN crop suffixes (all)
    stem = re.sub(                                  # the "titi" token anywhere
        r"(?i)(?<![a-z0-9])titi(?![a-z0-9])", "", stem
    )
    stem = re.sub(                                  # 2x / 3x / 4x / 6x markers
        r"(?<![A-Za-z0-9])\d+x(?![A-Za-z0-9])", "", stem, flags=re.IGNORECASE
    )
    stem = re.sub(r"(_\d{1,2})+$", "", stem)        # trailing _1, _2, ...
    stem = re.sub(r"[_-]{2,}", "_", stem).strip("_-")
    return stem or "photo"


def sidecar_for(image: Path) -> Path | None:
    """darktable writes ``foo.png.xmp``; some setups use ``foo.xmp``."""
    full = image.parent / (image.name + ".xmp")
    if full.exists():
        return full
    base = image.with_suffix(".xmp")
    if base.exists():
        return base
    return None


def read_dates(sources: list[Path]) -> dict[Path, str]:
    """Return {source: 'YYYY-MM-DD'} for sources that carry a capture date."""
    if not sources:
        return {}
    try:
        result = subprocess.run(
            [
                "exiftool", "-json", "-dateFormat", "%Y-%m-%d",
                *(f"-{tag}" for tag in DATE_TAGS),
                *map(str, sources),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        records = json.loads(result.stdout or "[]")
    except (FileNotFoundError, json.JSONDecodeError) as error:
        logging.warning("exiftool unavailable/unreadable (%s); all files undated", error)
        return {}

    dates: dict[Path, str] = {}
    for record in records:
        source = Path(record.get("SourceFile", "")).resolve()
        for tag in DATE_TAGS:
            value = record.get(tag)
            if value and re.match(r"\d{4}-\d{2}-\d{2}", str(value)):
                dates[source] = str(value)[:10]
                break
    return dates


def target_sidecar(sidecar: Path, image: Path, new_image: Path) -> Path:
    """Map a sidecar to the renamed image, preserving its naming form."""
    if sidecar.name == image.name + ".xmp":          # foo.png.xmp
        return new_image.parent / (new_image.name + ".xmp")
    return new_image.with_suffix(".xmp")             # foo.xmp


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    folder = args.input_path.expanduser().resolve()
    if not folder.is_dir():
        logging.error("Not a directory: %s", folder)
        return 2
    if args.fallback_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.fallback_date):
        logging.error("--fallback-date must be YYYY-MM-DD")
        return 2

    images = find_input_files(folder, args.recursive)
    if not images:
        logging.error("No supported images in %s", folder)
        return 1

    # Read the date from each image's sidecar when present, else the image.
    sidecars = {image: sidecar_for(image) for image in images}
    sources = {image: (sidecars[image] or image) for image in images}
    dates_by_source = read_dates(list({s.resolve() for s in sources.values()}))

    fallback = args.fallback_date or PLACEHOLDER_DATE
    date_of = {
        image: dates_by_source.get(source.resolve(), fallback)
        for image, source in sources.items()
    }
    undated = sum(1 for image in images if date_of[image] == fallback)

    # Number in ascending capture-date order (then name, for stable ties).
    ordered = sorted(images, key=lambda image: (date_of[image], image.name.casefold()))

    planned: list[tuple[Path, Path]] = []
    taken: set[Path] = set()

    def claim(source: Path, target: Path) -> bool:
        if target == source:
            return True
        if target.exists() or target in taken:
            logging.warning("skip %s -> %s (target exists)", source.name, target.name)
            return False
        taken.add(target)
        planned.append((source, target))
        return True

    for offset, image in enumerate(ordered):
        index = args.start + offset
        name = (
            f"{date_of[image]}_{args.name}_p{index:03d}"
            f"__{clean_original(image.stem)}{image.suffix.lower()}"
        )
        new_image = image.with_name(name)
        if not claim(image, new_image):
            continue
        sidecar = sidecars[image]
        if sidecar is not None:
            claim(sidecar, target_sidecar(sidecar, image, new_image))

    for source, target in planned:
        logging.info("%s  ->  %s", source.name, target.name)

    if undated:
        logging.info(
            "\n%d/%d image(s) had no capture date; used %s.",
            undated, len(images), fallback,
        )

    if not args.apply:
        logging.info("\nDry run. %d rename(s) planned; pass --apply to do it.", len(planned))
        return 0

    for source, target in planned:
        source.rename(target)
    logging.info("\nRenamed %d file(s).", len(planned))
    return 0


if __name__ == "__main__":
    sys.exit(main())
