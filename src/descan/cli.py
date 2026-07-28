#!/usr/bin/env python3
"""Unified ``descan`` command dispatching to the subtools."""

from __future__ import annotations

import importlib
import sys
from importlib.metadata import PackageNotFoundError, version

SUBCOMMANDS = {
    "split": "descan.ingest",
    "trim": "descan.autocrop_review",
    "crop": "descan.manualcrop",
    "rename": "descan.rename_photos",
    "montage": "descan.montage",
}

USAGE = (
    "usage: descan {" + "|".join(SUBCOMMANDS) + "} ...\n"
    "       descan --version\n\n"
    "Run `descan <command> --help` for a command's options."
)


def _version() -> str:
    try:
        return version("descan")
    except PackageNotFoundError:
        return "0+unknown"


def main() -> int:
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"descan {_version()}")
        return 0

    command = argv[0]
    if command not in SUBCOMMANDS:
        print(f"descan: unknown command {command!r}\n\n{USAGE}", file=sys.stderr)
        return 2

    module = importlib.import_module(SUBCOMMANDS[command])
    sys.argv = [f"descan {command}", *argv[1:]]  # let the subtool parse the rest
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
