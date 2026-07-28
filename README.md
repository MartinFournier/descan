# descan

Split the individual photos out of flatbed scans — one auto-oriented PNG per
photo. One scan usually holds several loose prints; `descan` writes each to its
own file.

## Use at your own risk

Vibe coded, tuned against one person's scans, provided as-is (see
[LICENSE](LICENSE)). It can crop wrong, miss photos, or mis-orient them. **Keep
your originals; review output before deleting anything.** Input scans are never
modified, but `manualcrop.py --replace` and `autocrop_review.py` overwrite/delete
*crops* on purpose.

## Pipeline

Run in order — first is automatic, the rest are quick manual passes:

1. **`ingest.py`** — split each scan into one PNG per photo.
2. **`autocrop_review.py`** — trim leftover white lid margins.
3. **`manualcrop.py`** — hand-fix misses / bad splits.
4. **`rename_photos.py`** — dated, ordered filenames.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install "opencv-python>=5" numpy
```

YuNet model ships in `assets/`. `rename_photos.py` also needs `exiftool` on PATH.

## `ingest.py` — split

```bash
python ingest.py ~/Scans ~/Out --dry-run --debug   # preview + overlays in Out/debug
python ingest.py ~/Scans ~/Out                     # write crops
```

Each photo is written whole (not cropped/deskewed) as `<scan>_pNN.png`, then
auto-oriented via face detection. Eyeball `--debug` overlays before a real run.

| Flag | Purpose |
|------|---------|
| `--dry-run` | Report only, write nothing. |
| `--debug` | Detection overlays to `<out>/debug/`. |
| `--min-area F` | Smallest photo, fraction of scan (`0.012`). Lower catches tiny prints. |
| `--max-photos N` | Cap per scan (`8`). |
| `--margin F` | Border kept around each crop (`0.003`). |
| `--no-auto-orient` | Skip rotation. |
| `--always-landscape` | Portrait → landscape when no face decides. |
| `--processing-size PX` | Detection resolution (`1200`); output is full-res. |
| `--overwrite` / `--recursive` / `--verbose` | As named. |

## `autocrop_review.py` — trim white margins

```bash
python autocrop_review.py ~/Out
```

Shows only crops it would change (original | trim). **y** accept, **n** skip,
**f** flag, **q** quit. Flagged files move to `to-split/` for the next step.

## `manualcrop.py` — hand-fix

```bash
python manualcrop.py ~/Scans/one.png ~/Out              # draw boxes on a scan
python manualcrop.py ~/Out/to-split ~/Out --replace     # re-split bad crops in place
```

Drag boxes (full-window crosshair for precise corners). **u** undo, **c** clear,
**ENTER**/**ESC** done. Saves `<stem>_pNN.png`, auto-oriented, numbered after
existing crops. `--replace`: original deleted, replaced by its `_pNN` pieces
(skipped files untouched). Dir arg walks each image; no box = skip.

## `rename_photos.py` — rename with dates

```bash
python rename_photos.py ~/Out                                   # dry run
python rename_photos.py ~/Out --apply --name Denise             # do it
python rename_photos.py ~/Out --apply --fallback-date 1994-01-01
```

Renames to `YYYY-MM-DD__<Name>_pNNN__<clean>.ext`, numbered `pNNN` in ascending
capture-date order. Date from each photo's `.xmp` sidecar (or EXIF) via
`exiftool`; the sidecar is renamed alongside. Name is cleaned (`titi`, `_pNN`,
`2x/3x`, `_1` stripped: `titi_3x_famille_p02` → `famille`). Undated files use
`--fallback-date` (or `0000-00-00`) and are reported. Re-running is idempotent
and re-sequences after adds/deletes.

## Development

```bash
pip install ruff
./gate.sh            # ruff format --check + ruff check (run before pushing)
```

Detection is classical CV tuned for light prints on a light lid. Tuning and the
full engineering rationale are in [AGENTS.md](AGENTS.md).
