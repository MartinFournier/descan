"""Shared argparse fragments used by more than one command."""

from __future__ import annotations

import argparse


def add_recursive(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories.",
    )


def add_png_compression(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(0, 10),
        default=6,
        metavar="0-9",
        help="PNG compression level (lossless; higher is smaller/slower). Default: 6.",
    )
