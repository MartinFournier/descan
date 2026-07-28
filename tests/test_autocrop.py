import numpy as np

from descan import autocrop_review


def test_trim_box_peels_white_side():
    # 100 tall x 200 wide, white; content only in the left third.
    image = np.full((100, 200, 3), 250, np.uint8)
    image[:, 0:60] = 40
    x0, y0, x1, y1 = autocrop_review.trim_box(
        image, lid_lightness=225.0, edge_fraction=0.9, margin_fraction=0.0
    )
    assert x0 == 0
    assert 55 <= x1 <= 70  # trimmed to just past the content edge
    assert (y0, y1) == (0, 100)


def test_trim_box_all_lid_is_noop():
    image = np.full((50, 50, 3), 255, np.uint8)
    assert autocrop_review.trim_box(image, 225.0, 0.9, 0.0) == (0, 0, 50, 50)
