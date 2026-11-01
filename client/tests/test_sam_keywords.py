from types import SimpleNamespace

from atas.review.film_review import validate_dynamic_classes
from atas.world.keywords import with_people


def test_onstage_actors_always_get_the_person_class():
    shot = SimpleNamespace(entity_ids=("chen_mo",))
    assert with_people(("man", "umbrella"), shot) == ("person", "man", "umbrella")
    assert with_people(("person", "photo"), shot) == ("person", "photo")
    assert with_people((), SimpleNamespace(entity_ids=())) == ()


def test_distinguishing_prop_phrases_are_valid_classes():
    rows = validate_dynamic_classes([{"noun": "Book in hand", "evidence": "pulled from shelf frame 0"}])
    assert rows[0]["noun"] == "book in hand"
