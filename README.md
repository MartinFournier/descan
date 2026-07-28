# descan

Split the individual photos out of flatbed scans into one auto-oriented PNG per
photo. One scan usually holds several loose prints; `descan` writes each to its
own file.

## Pipeline

Run in order. The first is automatic, the rest are quick manual passes:

1. **`ingest.py`** splits each scan into one PNG per photo.
2. **`autocrop_review.py`** trims leftover white lid margins.
3. **`manualcrop.py`** hand-fixes misses and bad splits.
4. **`rename_photos.py`** writes dated, ordered filenames.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install "opencv-python>=5" numpy
```

The YuNet face model ships in `assets/`. `rename_photos.py` also needs
`exiftool` on PATH.

## `ingest.py` (split)

```bash
python src/ingest.py ~/Scans ~/Out --dry-run --debug   # preview, overlays in Out/debug
python src/ingest.py ~/Scans ~/Out                     # write crops
```

Each photo is written whole (not cropped or deskewed) as `<scan>_pNN.png`, then
auto-oriented via face detection. Eyeball the `--debug` overlays before a real
run.

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

## `autocrop_review.py` (trim white margins)

```bash
python src/autocrop_review.py ~/Out
```

Shows only crops it would change, original beside trim. **y** accept, **n**
skip, **f** flag, **q** quit. Flagged files move to `to-split/` for the next
step.

## `manualcrop.py` (hand-fix)

```bash
python src/manualcrop.py ~/Scans/one.png ~/Out              # draw boxes on a scan
python src/manualcrop.py ~/Out/to-split ~/Out --replace     # re-split bad crops in place
```

Drag boxes; a full-window crosshair gives precise corners. **u** undo, **c**
clear, **ENTER**/**ESC** done. Saves `<stem>_pNN.png`, auto-oriented, numbered
after existing crops. `--replace` deletes the original and writes its `_pNN`
pieces (skipped files untouched). A directory argument walks each image; no box
means skip.

## `rename_photos.py` (rename with dates)

```bash
python src/rename_photos.py ~/Out                                   # dry run
python src/rename_photos.py ~/Out --apply --name Grandma            # do it
python src/rename_photos.py ~/Out --apply --fallback-date 1994-01-01
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
pip install ruff
./scripts/gate.sh            # ruff format --check + ruff check (run before pushing)
```

Detection is classical CV tuned for light prints on a light lid. Tuning and the
full engineering rationale live in [AGENTS.md](AGENTS.md).

## Credits

Face detection uses the YuNet model (`assets/face_detection_yunet_2023mar.onnx`)
from [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
MIT licensed, copyright Shiqi Yu (full text in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Use at your own risk

Vibe coded, tuned against one person's scans, provided as-is (see
[LICENSE](LICENSE)). It can crop wrong, miss photos, or mis-orient them. Keep
your originals and review output before deleting anything. Input scans are never
modified, but `manualcrop.py --replace` and `autocrop_review.py` overwrite or
delete *crops* on purpose.
