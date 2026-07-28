import numpy as np

from descan import ingest


def test_fill_holes_fills_enclosed_region():
    mask = np.zeros((7, 7), np.uint8)
    mask[1:6, 1:6] = 255
    mask[2:5, 2:5] = 0  # a hole enclosed by foreground
    filled = ingest.fill_holes(mask)
    assert filled[3, 3] == 255  # hole is filled
    assert filled[0, 0] == 0  # background stays background


def test_fill_holes_keeps_border_open_gap():
    # Two blocks with a gap that runs to the image edge must not merge.
    mask = np.zeros((10, 10), np.uint8)
    mask[:, 0:4] = 255
    mask[:, 6:10] = 255
    filled = ingest.fill_holes(mask)
    assert filled[5, 5] == 0  # the open gap stays background


def test_merge_overlapping_boxes():
    boxes = [(0, 0, 10, 10), (5, 5, 15, 15), (100, 100, 120, 120)]
    merged = ingest.merge_overlapping_boxes(boxes, overlap_fraction=0.1)
    assert len(merged) == 2
    assert (0, 0, 15, 15) in merged  # the overlapping pair unioned
    assert (100, 100, 120, 120) in merged  # the far box untouched


def test_crop_bounding_box():
    image = np.zeros((100, 200, 3), np.uint8)
    points = np.array([[20, 10], [60, 10], [60, 40], [20, 40]], np.float32)
    crop = ingest.crop_bounding_box(image, points, margin_pixels=0)
    assert crop.shape[:2] == (30, 40)  # (y1-y0, x1-x0)


def _scan_with_two_photos():
    # White lid, two well-separated dark rectangles.
    image = np.full((400, 600, 3), 245, np.uint8)
    image[60:340, 40:240] = 30
    image[60:340, 360:560] = 30
    return image


def test_find_photo_detections_counts_two():
    detections, _mask, _scale = ingest.find_photo_detections(
        _scan_with_two_photos(),
        minimum_area_fraction=0.02,
        maximum_photos=8,
        processing_size=1200,
    )
    assert len(detections) == 2
    # Left photo comes first in reading order.
    assert detections[0].centre_x < detections[1].centre_x
