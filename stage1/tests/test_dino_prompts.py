import numpy as np

from methods.dino_sam3.prompts import K_CANDIDATES, LONG_SIDE, NMS_PATCHES, PATCH, peaks, resized_hw


def test_resized_hw_multiples_of_patch_under_long_side():
    rh, rw = resized_hw(1080, 1920)
    assert rh % PATCH == 0 and rw % PATCH == 0 and max(rh, rw) <= LONG_SIDE and rw > rh


def test_peaks_returns_separated_maxima_highest_first_with_nms():
    sim = np.zeros((100, 100), np.float32)
    sim[20, 20] = 0.9
    sim[22, 22] = 0.85          # within the NMS radius of the first -> suppressed
    sim[70, 60] = 0.7
    out = peaks(sim, radius_px=NMS_PATCHES * PATCH, k=K_CANDIDATES)
    assert [p["point"] for p in out[:2]] == [[20, 20], [60, 70]]
    assert out[0]["cos"] > out[1]["cos"]
    assert all(abs(p["point"][0] - 20) > 5 or abs(p["point"][1] - 20) > 5 for p in out[1:])


def test_pick_first_candidate_at_or_above_tau():
    from methods.dino_sam3.run import pick
    assert pick([{"point": [1, 1], "cos": 0.6}, {"point": [2, 2], "cos": 0.4}])["point"] == [1, 1]
    assert pick([{"point": [1, 1], "cos": 0.49}, {"point": [2, 2], "cos": 0.1}]) is None
    assert pick([{"point": [3, 3], "cos": 0.5}])["point"] == [3, 3]
    assert pick([]) is None
