import json

from geomem.folds import eval_scenes, make_folds, train_scenes

SCENES = ["AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast", "CBABasketball", "Carpark",
          "CoffeeMartini", "Dog", "Fencing", "FlameSteak", "Frog", "MATF", "MartialArts", "Painter", "PoznanStreet", "Welder"]
HEMI = {"AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Dog", "Welder"}


def test_folds_partition():
    f = make_folds(SCENES)
    assert set(f) == set(SCENES)
    assert len({f[s] for s in HEMI}) == 4
    sizes = sorted([sum(1 for s in f if f[s] == k) for k in range(5)], reverse=True)
    assert sizes == [4, 4, 3, 3, 3]
    for k in range(5):
        assert set(train_scenes(f, k)) | set(eval_scenes(f, k)) == set(SCENES)
        assert not set(train_scenes(f, k)) & set(eval_scenes(f, k))


def test_folds_json_matches_generator():
    assert json.load(open("geomem/folds.json")) == make_folds(SCENES)
