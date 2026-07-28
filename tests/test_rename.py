import rename_photos as R


def test_clean_original_basic():
    assert R.clean_original("3x_beach_p02") == "beach"
    assert R.clean_original("house_p03_p01") == "house"  # repeated crop suffix
    assert R.clean_original("oldie_x6_2") == "oldie"  # xN marker + trailing index
    assert R.clean_original("voilier2") == "voilier"  # glued trailing number


def test_clean_original_keeps_years_and_leading_numbers():
    assert R.clean_original("noel_2001_p04") == "noel_2001"  # 4-digit year kept
    assert R.clean_original("80ans_3") == "80ans"  # leading number kept


def test_clean_original_strip_tokens():
    assert R.clean_original("titi_3x_beach_p02", ("titi",)) == "beach"
    assert R.clean_original("a_titi_b", ("titi",)) == "a_b"


def test_clean_original_empty_falls_back():
    assert R.clean_original("3x_p01") == "photo"


def test_split_applied_prefix_roundtrip():
    date, base = R.split_applied_prefix("2012-01-01__Person_p084__beach", "Person")
    assert date == "2012-01-01"
    assert base == "beach"


def test_split_applied_prefix_none_when_not_applied():
    date, base = R.split_applied_prefix("titi_beach_p01", "Person")
    assert date is None
    assert base == "titi_beach_p01"


def test_target_sidecar_forms(tmp_path):
    image = tmp_path / "old.png"
    new_image = tmp_path / "new.png"
    full = tmp_path / "old.png.xmp"  # darktable default
    base = tmp_path / "old.xmp"
    assert R.target_sidecar(full, image, new_image).name == "new.png.xmp"
    assert R.target_sidecar(base, image, new_image).name == "new.xmp"
