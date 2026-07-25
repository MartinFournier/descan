# ingest-scans

Detect, deskew, crop, orient, and split the individual printed photos out of
flatbed-scanner images. A single scan usually holds several loose prints laid on
the glass; this produces one clean PNG per photo.

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

Drag a box around each photo, **ENTER** after each, **ESC** when done. Each box
is saved as `<stem>_pNN.png`, auto-oriented the same way as `ingest.py`. New
crops are numbered *after* any existing ones for that scan, so they add to the
automatic output rather than replacing it (use `--overwrite` to renumber from
p01). A directory argument walks every scan in turn.

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
