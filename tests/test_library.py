from tools.library import normalize, slug


def test_unicode_names_and_long_labels_are_stable():
    assert normalize("Álbum") == normalize("A\u0301lbum")
    assert slug("Another Brick in the Wall, Pt. 2") == "another_brick_in_the_wall_pt_2"
    assert len(slug("a" * 100)) == 64
    assert slug("a" * 100) != slug("a" * 99 + "b")
