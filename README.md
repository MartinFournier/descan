# descan

[![CI](https://github.com/MartinFournier/descan/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinFournier/descan/actions/workflows/ci.yml)

Split the individual photos out of flatbed scans into one auto-oriented PNG per
photo. One scan usually holds several loose prints; `descan` writes each to its
own file.

## Pipeline

Run in order. The first is automatic, the rest are quick manual passes:

1. **`descan-split`** splits each scan into one PNG per photo.
2. **`descan-trim`** trims leftover white lid margins.
3. **`descan-crop`** hand-fixes misses and bad splits.
4. **`descan-rename`** writes dated, ordered filenames.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

Installs the `descan-split`, `descan-trim`, `descan-crop`, `descan-rename`, and
`descan-montage` commands, plus a unified `descan <command>` dispatcher
(`descan --version`, `descan --help`). Python 3.14 (see `.python-version`); the
YuNet model ships inside the package. `descan-rename` also needs the `exiftool`
system package on PATH.

## `descan-split`

```bash
descan-split ~/Scans ~/Out --dry-run --debug   # preview, overlays in Out/debug
descan-split ~/Scans ~/Out                     # write crops
```

Each photo is written whole (not cropped or deskewed) as `<scan>_pNN.png`, then
auto-oriented via face detection. Eyeball the `--debug` overlays before a real
run; `descan-montage ~/Out/debug` tiles them into one grid to scan at a glance.

| Flag | Purpose |
|------|---------|
| `--dry-run` | Report only, write nothing. |
| `--debug` | Detection overlays to `<out>/debug/`. |
| `--min-area F` | Smallest photo, fraction of scan (`0.012`). Lower catches tiny prints. |
| `--max-photos N` | Cap per scan (`8`). |
| `--margin F` | Border kept around each crop (`0.003`). |
| `--no-auto-orient` | Skip rotation. |
| `--always-landscape` | Portrait to landscape when no face decides. |
| `--processing-size PX` | Detection resolution (`1200`); output is full-res. |
| `--overwrite` / `--recursive` / `--verbose` | As named. |

## `descan-trim`

```bash
descan-trim ~/Out
```

Shows only crops it would change, original beside trim. **y** accept, **n**
skip, **f** flag, **q** quit. Flagged files move to `to-split/` for the next
step.

## `descan-crop`

```bash
descan-crop ~/Scans/one.png ~/Out              # draw boxes on a scan
descan-crop ~/Out/to-split ~/Out --replace     # re-split bad crops in place
```

Drag boxes; a full-window crosshair gives precise corners. **u** undo, **c**
clear, **ENTER**/**ESC** done. Saves `<stem>_pNN.png`, auto-oriented, numbered
after existing crops. `--replace` deletes the original and writes its `_pNN`
pieces (skipped files untouched). A directory argument walks each image; no box
means skip.

## `descan-rename`

```bash
descan-rename ~/Out                                   # dry run
descan-rename ~/Out --apply --name Grandma            # do it
descan-rename ~/Out --apply --fallback-date 1994-01-01
```

Renames to `YYYY-MM-DD__<Name>_pNNN__<clean>.ext`, numbered `pNNN` in ascending
capture-date order. Date comes from each photo's `.xmp` sidecar (or EXIF) via
`exiftool`, and the sidecar is renamed alongside. The name is cleaned: crop
suffixes (`_pNN`), sheet markers (`2x/3x`), and trailing `_1` are stripped, so
`3x_beach_p02` becomes `beach`; `--strip TOKEN` removes extra tokens (e.g.
`--strip titi`). Undated files use `--fallback-date` (or `0000-00-00`) and are
reported. Re-running is idempotent and re-sequences after adds or deletes.

## Development

```bash
pip install -e ".[dev]"
./scripts/gate.sh            # ruff format --check + ruff check (run before pushing)
pytest                       # tests
```

Detection is classical CV tuned for light prints on a light lid. Tuning and the
full engineering rationale live in [AGENTS.md](AGENTS.md).

## Credits

Face detection uses the YuNet model
(`src/descan/assets/face_detection_yunet_2023mar.onnx`) from
[opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
MIT licensed, copyright Shiqi Yu (full text in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Use at your own risk

Vibe coded, tuned against one person's scans, provided as-is (see
[LICENSE](LICENSE)). It can crop wrong, miss photos, or mis-orient them. Keep
your originals and review output before deleting anything. Input scans are never
modified, but `descan-crop --replace` and `descan-trim` overwrite or delete
*crops* on purpose.
