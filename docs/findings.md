# Findings & rejected approaches

The "why" behind `descan`'s detection, kept out of `AGENTS.md`. AGENTS has the
current design; this is history and dead ends.

## Iteration history

- **v1 (original, ChatGPT-generated):** background-colour subtraction. On white
  prints against a near-white lid the colour difference is ~zero, so it detected
  almost nothing (31 photos across 84 scans; 53 scans returned nothing).
- **v2:** foreground from Canny edges + interior gradient, dilate/close/open,
  then `minAreaRect` + perspective warp. On irregular/partial components
  `minAreaRect` produced skewed diamonds over sub-regions of one photo (a face,
  a patch of water) → 2–3 rotated slivers per print. Dropped: flatbed prints are
  near axis-aligned, so upright `boundingRect` is correct and simpler.
- **v3 (current):** not-background mask → `fill_holes` → `boundingRect` →
  overlap-merge → expand-to-lid → content-bridge merge. ~220 photos across the
  84 scans.

## fill_holes seed bug

v2 avoided `fill_holes` for fear of welding pages. The real welds came from (a)
an over-aggressive `MORPH_CLOSE` ringing the whole page before filling, and (b)
flooding from `(0,0)` when a photo touches that corner. Fix: pad a background
frame and seed the flood there. Inter-photo gaps run to the border, so the flood
reaches them and they stay open.

## Why expand + content-merge are separate from the mask

The mask separates and counts photos well but hugs content, clipping white
borders and pale regions (it cut pale-walled interior shots and bright-sky beach
shots in half). `expand_boxes_to_lid` recovers the true extent using whiteness
only as a *stop* signal: it stops at the white gap, so it can never merge.

Using flatness as a *positive* foreground signal instead welds everything,
because lid banding reads as texture (tried it: whole scan → 1 box).

`merge_content_bridged_boxes` is deliberately strict (union `mean(is_lid) <
0.10`). It fires on clearly dark over-splits (a dark indoor shot split by a pale
wall) but abstains when a photo has large bright regions that mimic a gap (a
big-sky shot stays split), and never welds pale/faded prints (welding faded sepia
prints to one box lost photos). Bias to abstain: losing a photo is worse than an
over-split, which is just an extra file to discard.

## Dead code removed in the rewrite

`sample_background_colour`, `contour_to_quad`, `points_are_reasonable`,
`polygon_iou`, and the `--background-threshold` flag / `background_threshold`
param (a leftover of the v1 colour-subtraction mask).
