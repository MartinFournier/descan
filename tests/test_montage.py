import numpy as np

from descan import cli, montage


def test_build_montage_grid_shape(tmp_path):
    import cv2

    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        cv2.imwrite(str(p), np.full((40, 60, 3), 200, np.uint8))
        paths.append(p)

    grid = montage.build_montage(paths, columns=2, tile_height=30)
    # 3 tiles at 2 columns -> 2 rows; each tile is 30px tall.
    assert grid.shape[0] == 60
    assert grid.ndim == 3


def test_cli_lists_expected_subcommands():
    assert set(cli.SUBCOMMANDS) == {"split", "trim", "crop", "rename", "montage"}


def test_cli_version(capsys):
    import sys

    old = sys.argv
    try:
        sys.argv = ["descan", "--version"]
        assert cli.main() == 0
    finally:
        sys.argv = old
    assert "descan" in capsys.readouterr().out
