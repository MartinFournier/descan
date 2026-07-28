"""Photo detection: build a per-photo mask and turn it into crop boxes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from descan.imageio import resize_for_detection


@dataclass
class Detection:
    """Detected photo represented in full-resolution source coordinates."""

    points: np.ndarray
    area: float
    centre_x: float
    centre_y: float


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill enclosed holes in a binary mask, solidifying each region.

    Flood the background inward from a guaranteed-background border (a padded
    frame, since a photo may touch the real image corner), then whatever the
    flood could not reach is an interior hole and gets turned on. This is what
    makes each photo a single solid blob even when its interior contains light
    sky or white clothing.
    """
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, np.zeros((0, 0), np.uint8), (0, 0), 255)
    background = flooded[1:-1, 1:-1]
    return mask | cv2.bitwise_not(background)


def build_detection_mask(image: np.ndarray) -> np.ndarray:
    """
    Build a mask whose connected components are individual printed photos.

    Family-album scans are white-bordered prints on a near-white scanner lid.
    The approach is deliberately simple: mark every pixel that is *not*
    background (darker than the lid, or coloured), fill each photo solid so its
    light interior regions are included, then open away thin links and speckle.
    Each surviving component is one whole photo.

    The returned mask is 8-bit, 255 on photo regions and 0 on background.
    """
    height, width = image.shape[:2]
    short_side = min(height, width)

    # Denoise to suppress low-amplitude scanner banding.
    denoised = cv2.bilateralFilter(image, 9, 50, 50)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[:, :, 0]
    chroma = np.abs(lab[:, :, 1] - 128.0) + np.abs(lab[:, :, 2] - 128.0)

    # Background lightness estimated from the scan border, which is almost
    # always lid rather than photo.
    border = np.concatenate(
        [
            lightness[:3].ravel(),
            lightness[-3:].ravel(),
            lightness[:, :3].ravel(),
            lightness[:, -3:].ravel(),
        ]
    )
    background_lightness = float(np.median(border))

    # Foreground: noticeably darker than the lid, or noticeably coloured.
    foreground = (((background_lightness - lightness) > 12.0) | (chroma > 10.0)).astype(
        np.uint8
    ) * 255

    # Close small speckle gaps, fill each photo solid, then open to drop thin
    # bridges between adjacent prints and isolated noise.
    close_size = max(3, int(short_side * 0.010)) | 1
    foreground = cv2.morphologyEx(  # type: ignore[assignment]
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    foreground = fill_holes(foreground)

    open_size = max(3, int(short_side * 0.012)) | 1
    foreground = cv2.morphologyEx(  # type: ignore[assignment]
        foreground,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )

    return foreground


def quad_area(points: np.ndarray) -> float:
    return abs(float(cv2.contourArea(points.astype(np.float32))))


def find_photo_detections(
    image: np.ndarray,
    minimum_area_fraction: float,
    maximum_photos: int,
    processing_size: int,
) -> tuple[list[Detection], np.ndarray, float]:
    """Detect candidate printed-photo rectangles."""
    working, scale = resize_for_detection(image, processing_size)
    working_height, working_width = working.shape[:2]
    image_area = float(working_width * working_height)

    mask = build_detection_mask(working)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    # Axis-aligned bounding boxes, in working-image coordinates. We deliberately
    # avoid minAreaRect: fitting a rotated rectangle to an irregular or partial
    # component produces skewed slivers over sub-regions of a single photo. The
    # goal here is only to split the scan into whole, uncropped photos, so an
    # upright box that keeps the full print (border included) is what we want.
    boxes: list[tuple[float, float, float, float]] = []

    for label in range(1, component_count):
        component_area = float(stats[label, cv2.CC_STAT_AREA])
        if component_area / image_area < minimum_area_fraction * 0.5:
            continue

        x = float(stats[label, cv2.CC_STAT_LEFT])
        y = float(stats[label, cv2.CC_STAT_TOP])
        w = float(stats[label, cv2.CC_STAT_WIDTH])
        h = float(stats[label, cv2.CC_STAT_HEIGHT])

        if w < 8 or h < 8:
            continue

        area_fraction = (w * h) / image_area
        aspect_ratio = max(w, h) / min(w, h)

        # A photo is a plausibly-proportioned box. The upper area bound also
        # rejects the near-full-page blob that appears when detection fails.
        if not (minimum_area_fraction < area_fraction < 0.94 and aspect_ratio < 6.5):
            continue

        boxes.append((x, y, x + w, y + h))

    # Merge overlapping boxes into one per photo. This collapses the skewed
    # slivers and sky/subject splits that previously became separate detections.
    boxes = merge_overlapping_boxes(boxes, overlap_fraction=0.12)

    # The mask hugs the coloured/dark content, so the box clips a photo's white
    # border and pale regions (sky, white clothing). Grow each box outward until
    # it reaches the flat white lid or a neighbouring photo, recovering the whole
    # print without merging separate ones.
    boxes = expand_boxes_to_lid(working, boxes)

    # A photo with a low-content band running through it (plain sand, a pale
    # wall) splits into adjacent boxes. Rejoin boxes whose shared border is photo
    # content rather than white lid; separate prints have a lid gap between them.
    boxes = merge_content_bridged_boxes(working, boxes)

    candidates: list[Detection] = []
    for x0, y0, x1, y1 in boxes:
        points = (
            np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.float32,
            )
            / scale
        )
        centre = points.mean(axis=0)
        candidates.append(
            Detection(
                points=points,
                area=quad_area(points),
                centre_x=float(centre[0]),
                centre_y=float(centre[1]),
            )
        )

    candidates.sort(key=lambda item: item.area, reverse=True)
    unique = candidates[:maximum_photos]

    # Reading order: cluster approximately into rows, then left to right.
    if unique:
        typical_height = np.median(
            [
                max(
                    np.linalg.norm(item.points[3] - item.points[0]),
                    np.linalg.norm(item.points[2] - item.points[1]),
                )
                for item in unique
            ]
        )
        row_tolerance = max(30.0, float(typical_height) * 0.35)

        unique.sort(
            key=lambda item: (
                round(item.centre_y / row_tolerance),
                item.centre_x,
            )
        )

    return unique, mask, scale


def expand_boxes_to_lid(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """
    Grow each detection box outward until it meets the scanner lid or a neighbour.

    The detection mask keys on coloured/dark content, so a box hugs that content
    and clips the print's white border and pale regions (sky, white clothing).
    Each side is pushed out in small steps and stops when the strip just beyond
    is mostly flat white lid, when it reaches another box, or at the image edge.
    Because it stops at the white gap, separate prints are never merged.
    """
    if not boxes:
        return boxes

    height, width = image.shape[:2]
    lab = cv2.cvtColor(cv2.bilateralFilter(image, 9, 50, 50), cv2.COLOR_BGR2LAB).astype(
        np.float32
    )
    lightness = lab[:, :, 0]
    chroma = np.abs(lab[:, :, 1] - 128.0) + np.abs(lab[:, :, 2] - 128.0)

    # Lid reference: the scan is mostly lid, so a high lightness percentile is
    # a robust white estimate. "Lid" is bright and near-neutral.
    lid_reference = float(np.percentile(lightness, 92))
    is_lid = ((lightness > lid_reference - 12.0) & (chroma < 7.0)).astype(np.uint8)

    step = max(2, int(min(height, width) * 0.004))
    lid_strip_fraction = 0.80

    def blocked(strip: np.ndarray) -> bool:
        return strip.size > 0 and float(strip.mean()) > lid_strip_fraction

    expanded: list[tuple[float, float, float, float]] = []
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        others = [o for j, o in enumerate(boxes) if j != index]

        def hits_neighbour(nx0, ny0, nx1, ny1, others=others) -> bool:
            for ox0, oy0, ox1, oy1 in others:
                if not (ox1 <= nx0 or ox0 >= nx1 or oy1 <= ny0 or oy0 >= ny1):
                    return True
            return False

        while y0 - step >= 0 and not blocked(is_lid[y0 - step : y0, x0:x1]):
            if hits_neighbour(x0, y0 - step, x1, y0):
                break
            y0 -= step
        while y1 + step <= height and not blocked(is_lid[y1 : y1 + step, x0:x1]):
            if hits_neighbour(x0, y1, x1, y1 + step):
                break
            y1 += step
        while x0 - step >= 0 and not blocked(is_lid[y0:y1, x0 - step : x0]):
            if hits_neighbour(x0 - step, y0, x0, y1):
                break
            x0 -= step
        while x1 + step <= width and not blocked(is_lid[y0:y1, x1 : x1 + step]):
            if hits_neighbour(x1, y0, x1 + step, y1):
                break
            x1 += step

        expanded.append((float(x0), float(y0), float(x1), float(y1)))

    return expanded


def merge_content_bridged_boxes(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """
    Rejoin boxes that are two parts of one photo split by a low-content band.

    When a plain sand strip or a dark wall runs through a photo, the mask breaks
    it into adjacent boxes. For an adjacent pair, if the *whole union rectangle*
    is almost entirely photo content (very little white lid inside it) the two
    boxes are one photo and are merged. Two separate prints leave a white lid
    gap inside the union, so it is not merged.

    The union test is deliberately strict (a low lid fraction). It fires on
    clearly dark over-splits but abstains when a photo has large bright regions
    that could be confused with a gap, and it protects pale/faded prints — which
    read as lid-like — from ever being welded together (losing a photo is far
    worse than an over-split, which is just an extra file). Iterates to a fixed
    point.
    """
    if len(boxes) < 2:
        return boxes

    height, width = image.shape[:2]
    lab = cv2.cvtColor(cv2.bilateralFilter(image, 9, 50, 50), cv2.COLOR_BGR2LAB).astype(
        np.float32
    )
    lightness = lab[:, :, 0]
    chroma = np.abs(lab[:, :, 1] - 128.0) + np.abs(lab[:, :, 2] - 128.0)
    lid_reference = float(np.percentile(lightness, 92))
    is_lid = ((lightness > lid_reference - 12.0) & (chroma < 7.0)).astype(np.uint8)

    work: list[list[int]] = [[int(round(v)) for v in box] for box in boxes]

    changed = True
    while changed:
        changed = False
        for i in range(len(work)):
            for j in range(i + 1, len(work)):
                a, b = work[i], work[j]
                y_overlap = min(a[3], b[3]) - max(a[1], b[1])
                x_overlap = min(a[2], b[2]) - max(a[0], b[0])
                min_height = min(a[3] - a[1], b[3] - b[1])
                min_width = min(a[2] - a[0], b[2] - b[0])

                side_by_side = (
                    y_overlap > 0.3 * min_height
                    and (max(a[0], b[0]) - min(a[2], b[2])) < 0.15 * min_width
                )
                stacked = (
                    x_overlap > 0.3 * min_width
                    and (max(a[1], b[1]) - min(a[3], b[3])) < 0.15 * min_height
                )

                if not (side_by_side or stacked):
                    continue

                union = is_lid[
                    min(a[1], b[1]) : max(a[3], b[3]),
                    min(a[0], b[0]) : max(a[2], b[2]),
                ]
                if union.size and float(union.mean()) < 0.10:
                    work[i] = [
                        min(a[0], b[0]),
                        min(a[1], b[1]),
                        max(a[2], b[2]),
                        max(a[3], b[3]),
                    ]
                    work.pop(j)
                    changed = True
                    break
            if changed:
                break

    return [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in work]


def merge_overlapping_boxes(
    boxes: list[tuple[float, float, float, float]],
    overlap_fraction: float,
) -> list[tuple[float, float, float, float]]:
    """
    Union axis-aligned boxes that overlap by more than ``overlap_fraction`` of
    the smaller box, iterating to a fixed point.

    Fragments of one photo (a sliver detection, or a sky region split from its
    subject) overlap heavily and collapse into a single box, while genuinely
    separate prints — kept apart by the white gap between them — do not.
    """
    boxes = list(boxes)
    changed = True

    while changed:
        changed = False
        result: list[tuple[float, float, float, float]] = []

        for box in boxes:
            for index, other in enumerate(result):
                ix0 = max(box[0], other[0])
                iy0 = max(box[1], other[1])
                ix1 = min(box[2], other[2])
                iy1 = min(box[3], other[3])

                intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                area_box = (box[2] - box[0]) * (box[3] - box[1])
                area_other = (other[2] - other[0]) * (other[3] - other[1])
                smaller = min(area_box, area_other)

                if smaller > 0 and intersection > overlap_fraction * smaller:
                    result[index] = (
                        min(box[0], other[0]),
                        min(box[1], other[1]),
                        max(box[2], other[2]),
                        max(box[3], other[3]),
                    )
                    changed = True
                    break
            else:
                result.append(box)

        boxes = result

    return boxes


def crop_bounding_box(
    image: np.ndarray,
    points: np.ndarray,
    margin_pixels: float,
) -> np.ndarray:
    """
    Return the uncropped photo as an axis-aligned slice of the source image.

    The detected box is padded by ``margin_pixels`` and clipped to the image, so
    the whole print is kept. No deskew or perspective correction is applied;
    splitting the scan is the only job here.
    """
    height, width = image.shape[:2]

    x0 = int(np.floor(points[:, 0].min() - margin_pixels))
    y0 = int(np.floor(points[:, 1].min() - margin_pixels))
    x1 = int(np.ceil(points[:, 0].max() + margin_pixels))
    y1 = int(np.ceil(points[:, 1].max() + margin_pixels))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x1)
    y1 = min(height, y1)

    return image[y0:y1, x0:x1].copy()


def make_debug_image(
    image: np.ndarray,
    detections: Sequence[Detection],
    maximum_dimension: int = 2200,
) -> np.ndarray:
    """Draw detected photo boundaries and their output ordering."""
    debug, scale = resize_for_detection(image, maximum_dimension)

    for index, detection in enumerate(detections, start=1):
        points = np.round(detection.points * scale).astype(np.int32)
        cv2.polylines(
            debug,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=max(2, int(max(debug.shape[:2]) / 900)),
        )

        label_point = tuple(points[0])
        cv2.putText(
            debug,
            f"p{index:02d}",
            (label_point[0] + 8, label_point[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return debug
