# descan

Detect, deskew, crop, orient, and split the individual printed photos out of
flatbed-scanner images. A single scan usually holds several loose prints laid on
the glass; this produces one clean PNG per photo.

## Use at your own risk

This project was vibe coded — built quickly and iteratively with an AI
assistant, tuned against one person's scan collection. It is provided as-is,
with no guarantees (see [LICENSE](LICENSE)). It can crop wrong, miss photos, or
mis-orient them. **Keep your original scans; treat all output as disposable and
review it before deleting anything.** The tools never modify input scans, but
`manualcrop.py --replace` and `autocrop_review.py` do overwrite/delete crops on
purpose — read what each does first.

## What it does

For every image in the input directory:

1. **Detect** each printed photo. These albums are mostly white-bordered prints
   on a near-white scanner lid, so detection marks every pixel that is *not*
   background (darker than the lid, or coloured), fills each photo solid so its
   light interior is included, then opens away the thin links between adjacent
   prints. Each surviving connected component is one photo.
2. **Split, don't crop.** Each photo is emitted as an axis-aligned slice of the
   source — the whole print, white border and all. No deskew or perspective
   correction: the only job is to separate the prints. Overlapping detections
   (slivers, or a bright region split from its subject) are merged so each print
   is one file.
3. **Auto-orient** each split to 0/90/180/270° using the YuNet DNN face
   detector: it scores all four rotations and picks the one where faces read as
   upright.
4. **Write** a lossless PNG per photo, named `<scan-stem>_pNN.png`.

Input files are only ever read, never modified.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install "opencv-python>=5" numpy
```

The YuNet face model ships in `assets/`. No other data files are needed.

## Usage

```bash
# Preview what would be written (no files created), with annotated overlays:
.venv/bin/python ingest.py ~/Scans ~/Photos/split --dry-run --debug

# Do it for real:
.venv/bin/python ingest.py ~/Scans ~/Photos/split
```

Overlays land in `<output>/debug/` when `--debug` is set — always eyeball these
before a real run.

### Useful flags

| Flag | Purpose |
|------|---------|
| `--dry-run` | Detect and report, write nothing. |
| `--debug` | Write annotated detection overlays to `<output>/debug/`. |
| `--min-area FRACTION` | Smallest photo as a fraction of the scan (default `0.012`). Raise to drop small false positives; lower to catch tiny prints. |
| `--max-photos N` | Cap detections per scan (default `8`). |
| `--margin FRACTION` | Extra border kept around each crop (default `0.003`). |
| `--no-auto-orient` | Skip face-based rotation. |
| `--always-landscape` | Rotate portrait crops to landscape when no face decides it. |
| `--processing-size PX` | Detection working resolution (default `1200`); output is always full-res. |
| `--overwrite` | Replace existing output PNGs. |
| `--recursive` | Walk subdirectories. |
| `--verbose` | Extra logging. |

## Manual cropping

For the few photos the detector misses or mis-splits (very faded/low-contrast
prints, or a photo whose bright sky looks like a gap), draw the boxes by hand:

```bash
python manualcrop.py ~/Scans/titi_mariage.png ~/Photos/split
```

Drag a box around each photo (a thin full-window **crosshair** tracks the cursor
for pixel-accurate corners); **u** undoes the last box, **c** clears, **ENTER**
or **ESC** finishes the image. Each box is saved as `<stem>_pNN.png`,
auto-oriented the same way as `ingest.py`. New
crops are numbered *after* any existing ones for that scan, so they add to the
automatic output rather than replacing it (use `--overwrite` to renumber from
p01). A directory argument walks every image in turn (`[3/12] name`); drawing no
box leaves that file untouched.

**Re-splitting crops that still hold two photos** (`--replace`): point it at the
folder of crops, using the same folder as output. Draw a box around each real
photo; the original is deleted and replaced by its `_pNN` pieces. Skipped files
(no box drawn) are left as they are.

```bash
python manualcrop.py ~/Photos/to_fix ~/Photos/to_fix --replace
```

## Trimming leftover white margins

Some crops keep a large white block on one side where the box overshot into the
scanner lid. `autocrop_review.py` peels flat white lid margins off each edge and
lets you approve each suggestion:

```bash
python autocrop_review.py ~/Photos/split
```

Only crops it would actually change are shown — original on the left, trim on the
right. Keys: **y**/ENTER accept (overwrites the file), **n** skip, **f** flag for
manual cropping, **q** quit. Flagged files are moved into a `to-split/` subfolder,
ready to re-split by hand:

```bash
python manualcrop.py ~/Photos/split/to-split ~/Photos/split --replace
```

## Renaming with dates

Once dates are set (e.g. in darktable, which writes a `.xmp` sidecar per photo),
`rename_photos.py` renames files to
`YYYY-MM-DD__<Name>_pNNN__<cleaned-original>.ext`:

```bash
python rename_photos.py ~/Photos/split                 # dry run (default)
python rename_photos.py ~/Photos/split --apply          # do it
python rename_photos.py ~/Photos/split --name Denise --fallback-date 1994-01-01
```

- Date comes from each photo's `.xmp` sidecar (or the image's own EXIF), read via
  `exiftool`. Files are numbered `pNNN` in **ascending capture-date order**.
- The `.xmp` sidecar is renamed alongside its image so darktable keeps the edit
  history linked.
- The original name is cleaned: `titi` token removed, crop suffixes (`_pNN`,
  including repeated), sheet markers (`2x/3x/4x/6x`), and trailing split indices
  (`_1`, `_2`) stripped — `titi_3x_famille_p02` → `famille`.
- Images with no date get `--fallback-date` (or `0000-00-00`) and are reported,
  so you can date them and re-run.

## Development

Lint and format with [ruff](https://docs.astral.sh/ruff/) (config in
`pyproject.toml`):

```bash
pip install ruff
ruff format .      # format
ruff check --fix . # lint
```

## Tuning notes

Detection is classical CV, tuned for *light prints with white borders on a
light lid*. If a run under- or over-splits:

- **Photos missed** (very light / low-detail prints): lower `--min-area`.
- **One box spanning two touching prints**: the open-sever kernel in
  `build_detection_mask` (`open_size`) may be too small; touching prints are the
  hard case.
- **A photo split in two** (e.g. plain sky above a subject): usually re-merged by
  the overlap step in `find_photo_detections`; if not, it surfaces as two crops.

Expect ~95%+ of photos on album-style scans; a handful of edge cases still need a
manual pass. See `AGENTS.md` for the full engineering history and rationale.
