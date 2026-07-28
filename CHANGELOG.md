# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-07-28

Initial release.

### Added

- `descan-split`: detect and split each flatbed scan into one whole,
  auto-oriented PNG per photo (YuNet face orientation).
- `descan-trim`: review/apply auto-trim of leftover white scanner-lid margins.
- `descan-crop`: GUI to hand-draw crop boxes for missed or mis-split photos,
  with an in-place `--replace` re-split mode.
- `descan-rename`: date-prefixed, capture-date-ordered renaming from `.xmp`/EXIF
  via exiftool; idempotent and renumber-safe.
- `descan-montage`: tile a directory of images into one labelled grid.
- Unified `descan <command>` dispatcher and `descan --version`.
- Packaging, ruff lint/format gate, pytest suite, and CI.

[Unreleased]: https://github.com/MartinFournier/descan/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MartinFournier/descan/releases/tag/v0.1.0
