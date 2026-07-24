# ingest-scans

Detect, deskew, crop, orient, and split the individual printed photos out of
flatbed-scanner images. A single scan usually holds several loose prints laid on
the glass; this produces one clean PNG per photo.

## What it does

For every image in the input directory:

1. **Detect** each printed photo. These albums are mostly white-bordered prints
   on a near-white scanner lid, so detection keys on faint print-border edges
   plus a colour/darkness signal from the photo interior, then severs the thin
   bridges (drop shadows, scanner banding) that would weld neighbouring prints
   into one blob. Each surviving connected component is one photo.
2. **Deskew + perspective-crop** each photo (small rotations are corrected).
3. **Auto-orient** to 0/90/180/270° using the YuNet DNN face detector: it scores
   all four rotations and picks the one where faces read as upright.
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
