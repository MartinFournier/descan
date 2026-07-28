#!/usr/bin/env python3
"""
Rename photos to ``YYYY-MM-DD_<Name>_pNNN__<cleaned-original>.ext``.

The date is read from each photo's darktable ``.xmp`` sidecar (or the image's
own EXIF if there is no sidecar), via exiftool. Files are numbered ``pNNN`` in
ascending capture-date order. Each image's sidecar is renamed alongside it so
darktable keeps the edit history linked.

The original name is cleaned: crop suffixes (``_pNN``, including repeated ones
from re-splits), the ``2x/3x/4x/6x`` sheet markers, trailing split indices
(``_1``, ``_2`` ...), and any tokens passed with ``--strip`` are removed.

    3x_beach_p02.png (+ .xmp)   -> 1994-08-28__Person_p007__beach.png (+ .xmp)
    house_p03_p01.png           -> ...__Person_pNNN__house.png

Re-running is safe: an already-applied ``DATE__Name_pNNN__`` prefix is stripped
before rebuilding (no stacking), the date is re-read from the sidecar (a prior
in-name date is used only as a fallback), and ``pNNN`` is renumbered over the
current set, so adding or deleting files just re-sequences them.

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
    parser.add_argument(
        "--name", default="Person", help="Subject name. Default: Person."
    )
    parser.add_argument(
        "--strip",
        action="append",
        default=[],
        metavar="TOKEN",
        help="Remove this token from names, repeatable (e.g. --strip titi).",
    )
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


def split_applied_prefix(stem: str, name: str) -> tuple[str | None, str]:
    """
    Undo a prefix this script already applied, so re-runs are idempotent.

    ``2012-01-01__Person_p084__beach`` -> ("2012-01-01", "beach").
    Returns (date-or-None, remaining-stem).
    """
    match = re.match(
        rf"^(\d{{4}}-\d{{2}}-\d{{2}})__{re.escape(name)}_p\d+__(.*)$", stem
    )
    if match:
        return match.group(1), match.group(2)
    return None, stem


def clean_original(stem: str, strip_tokens: tuple[str, ...] = ()) -> str:
    """Strip crop suffixes, sheet markers, split indices, and any given tokens."""
    stem = re.sub(r"(?i)_p\d+", "", stem)  # _pNN crop suffixes (all)
    for token in strip_tokens:  # caller-supplied tokens (e.g. a nickname)
        stem = re.sub(rf"(?i)(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", "", stem)
    stem = re.sub(  # 2x / 3x / x6 sheet markers
        r"(?i)(?<![a-z0-9])(\d+x|x\d+)(?![a-z0-9])", "", stem
    )
    stem = re.sub(r"(_\d{1,2})+$", "", stem)  # trailing _1, _2, ...
    stem = re.sub(r"[_-]{2,}", "_", stem).strip("_-")
    stem = re.sub(r"(?<=[A-Za-z])\d{1,2}$", "", stem)  # voilier2 -> voilier
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
                "exiftool",
                "-json",
                "-dateFormat",
                "%Y-%m-%d",
                *(f"-{tag}" for tag in DATE_TAGS),
                *map(str, sources),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        records = json.loads(result.stdout or "[]")
    except (FileNotFoundError, json.JSONDecodeError) as error:
        logging.warning(
            "exiftool unavailable/unreadable (%s); all files undated", error
        )
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
    if sidecar.name == image.name + ".xmp":  # foo.png.xmp
        return new_image.parent / (new_image.name + ".xmp")
    return new_image.with_suffix(".xmp")  # foo.xmp


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    folder = args.input_path.expanduser().resolve()
    if not folder.is_dir():
        logging.error("Not a directory: %s", folder)
        return 2
    if args.fallback_date and not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", args.fallback_date
    ):
        logging.error("--fallback-date must be YYYY-MM-DD")
        return 2

    images = find_input_files(folder, args.recursive)
    if not images:
        logging.error("No supported images in %s", folder)
        return 1

    # Recover the base name and any prior date from a prefix we applied before,
    # so re-runs stay idempotent instead of stacking prefixes.
    applied = {image: split_applied_prefix(image.stem, args.name) for image in images}

    # Read the date from each image's sidecar when present, else the image.
    sidecars = {image: sidecar_for(image) for image in images}
    sources = {image: (sidecars[image] or image) for image in images}
    dates_by_source = read_dates(list({s.resolve() for s in sources.values()}))

    fallback = args.fallback_date or PLACEHOLDER_DATE

    def resolve_date(image: Path) -> str:
        # Metadata wins; then a date this script wrote before; then the fallback.
        return (
            dates_by_source.get(sources[image].resolve())
            or applied[image][0]
            or fallback
        )

    date_of = {image: resolve_date(image) for image in images}
    undated = sum(1 for image in images if date_of[image] == fallback)

    # Number in ascending capture-date order (then name, for stable ties).
    ordered = sorted(images, key=lambda image: (date_of[image], image.name.casefold()))

    # Any current file may be renamed, so a target colliding with one of these
    # is fine (it will be vacated); only an outside file is a real conflict.
    managed = {p.resolve() for image in images for p in (image, sidecars[image]) if p}

    planned: list[tuple[Path, Path]] = []
    taken: set[Path] = set()

    def claim(source: Path, target: Path) -> bool:
        if target == source:
            return True
        if target in taken or (target.exists() and target.resolve() not in managed):
            logging.warning("skip %s -> %s (target exists)", source.name, target.name)
            return False
        taken.add(target)
        planned.append((source, target))
        return True

    for offset, image in enumerate(ordered):
        index = args.start + offset
        base = clean_original(applied[image][1], tuple(args.strip))
        name = (
            f"{date_of[image]}__{args.name}_p{index:03d}__{base}{image.suffix.lower()}"
        )
        new_image = image.with_name(name)
        if not claim(image, new_image):
            continue
        sidecar = sidecars[image]
        if sidecar is not None:
            claim(sidecar, target_sidecar(sidecar, image, new_image))

    width = max((len(source.name) for source, _ in planned), default=0)
    for source, target in planned:
        logging.info("%s  ->  %s", source.name.ljust(width), target.name)

    if undated:
        logging.info(
            "\n%d/%d image(s) had no capture date; used %s.",
            undated,
            len(images),
            fallback,
        )

    if not args.apply:
        logging.info(
            "\nDry run. %d rename(s) planned; pass --apply to do it.", len(planned)
        )
        return 0

    # Two-phase via temp names so a renumber can't collide with a not-yet-moved
    # file (e.g. p084 -> p351 while p351 still exists under its old name).
    staged: list[tuple[Path, Path]] = []
    for i, (source, target) in enumerate(planned):
        temp = source.with_name(f".rename_tmp_{i}")
        source.rename(temp)
        staged.append((temp, target))
    for temp, target in staged:
        temp.rename(target)
    logging.info("\nRenamed %d file(s).", len(planned))
    return 0


if __name__ == "__main__":
    sys.exit(main())
