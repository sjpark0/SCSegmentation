"""T8: P5(RepairSeeds)의 선행 확인 두 건을 코드로 고정합니다.

ROADMAP Phase 3은 P5 구현 전에 "refine 분기가 같은 tracker state에 cond frame을 추가하는지"
로그로 확인하라고 적었습니다(심사자 두 명이 갈렸던 자리). 로그 한 번은 그때만 참이므로,
대신 그 성질을 테스트로 못박습니다. 두 번째는 문서 어디에도 없던 순서 제약입니다.

이 파일은 torch를 임포트하지 않습니다 — 소스를 읽어 확인합니다.
"""
import ast
import os
import re

import pytest

from conftest import REPO

PKG = os.path.join(REPO, "SCSam3", "demoSCSam3MVOpt")
TRACKERS = ["SCSam3VideoInference.py", "SCSam3VideoInferenceNewMem.py"]
RUNNER = os.path.join(REPO, "SCSam3", "runMVSeg.py")


def src(name):
    with open(os.path.join(PKG, name), encoding="utf-8") as fh:
        return fh.read()


# ------------------------------------------------------- 선행 확인 (i): refine 분기
@pytest.mark.parametrize("name", TRACKERS)
def test_stateless_refinement_defaults_off(name):
    """생성자 기본값이 False입니다."""
    tree = ast.parse(src(name))
    seen = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        for arg, default in zip(args.args[-len(args.defaults):] if args.defaults else [],
                                args.defaults):
            if arg.arg == "use_stateless_refinement":
                assert isinstance(default, ast.Constant) and default.value is False, \
                    f"{name}:{node.lineno} {node.name} defaults it to {ast.dump(default)}"
                seen += 1
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if arg.arg == "use_stateless_refinement":
                assert isinstance(default, ast.Constant) and default.value is False
                seen += 1
    assert seen >= 1, f"{name}: use_stateless_refinement 인자를 찾지 못했습니다"


def test_stateless_refinement_is_never_enabled():
    """저장소 어디에서도 True로 켜지 않습니다 — 그래서 '객체를 지우고 다시 넣는' 분기는
    우리 실행에서 죽은 코드이고, 살아 있는 경로는 obj_id로 기존 tracker state를 재사용합니다.
    P5의 재전파는 그 재사용 경로를 타므로 새 tracker state가 생기지 않습니다."""
    pat = re.compile(r"use_stateless_refinement\s*=\s*True")
    hits = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "hf_cache", "Data")]
        for f in files:
            if f.endswith((".py", ".yaml", ".yml", ".json")):
                p = os.path.join(root, f)
                try:
                    with open(p, encoding="utf-8", errors="ignore") as fh:
                        if pat.search(fh.read()):
                            hits.append(os.path.relpath(p, REPO))
                except OSError:
                    pass
    assert hits == [], f"켜는 곳이 생겼습니다: {hits}. P5의 선행 가정이 깨집니다."


@pytest.mark.parametrize("name", TRACKERS)
def test_refinement_branch_reuses_the_existing_state(name):
    """죽은 분기 바로 아래의 살아 있는 경로가 obj_id로 기존 상태를 찾아 씁니다."""
    s = src(name)
    assert "_get_tracker_inference_states_by_obj_ids" in s
    assert "# existing object, for refinement" in s


# ------------------------------------------------------- 선행 확인 (ii): 순서 제약
def test_repair_must_run_before_the_spatial_model_is_retired():
    """RepairSeeds()는 반드시 RetireSpatialPredictor() **앞**에 들어가야 합니다.

    러너의 순서는 PropagateAcrossViews -> RetireSpatialPredictor -> TrackForward입니다.
    폐기는 시점 예측기와 그 특징 캐시를 버리므로, 그 뒤에서 재전파를 시도하면 에러가 아니라
    **조용히 아무것도 고치지 않은 채** 통과할 수 있습니다. 이 테스트는 세 호출의 순서를
    고정해, 나중에 누가 삽입 지점을 옮기면 실패하게 합니다."""
    with open(RUNNER, encoding="utf-8") as fh:
        lines = fh.readlines()
    def line_of(needle):
        for i, l in enumerate(lines, 1):
            if needle in l and not l.strip().startswith("#"):
                return i
        raise AssertionError(f"{needle} 를 runMVSeg.py 에서 찾지 못했습니다")
    prop = line_of("sc.PropagateAcrossViews(")
    retire = line_of("sc.RetireSpatialPredictor()")
    track = line_of("sc.TrackForward(")
    assert prop < retire < track, (prop, retire, track)
    # 복구가 구현되면 그 호출도 이 구간 안에 있어야 합니다
    with open(RUNNER, encoding="utf-8") as fh:
        s = fh.read()
    if "RepairSeeds(" in s:
        repair = line_of("sc.RepairSeeds(")
        assert prop < repair < retire, (
            "RepairSeeds() 는 PropagateAcrossViews() 뒤, RetireSpatialPredictor() 앞이어야 "
            f"합니다 (지금 {prop} < {repair} < {retire} 가 아님)")


def test_spatial_start_stays_implicit():
    """재전파는 SPATIAL_START_IMPLICIT 를 켠 채로 돌아야 합니다.

    시작 시점을 명시하면 역방향 전파가 강등되어 Blocks 가 0.7484 -> 0.2827 로 무너진 전례가
    있습니다(investigations-closed 4번). 기본값이 바뀌면 이 테스트가 알려 줍니다."""
    with open(RUNNER, encoding="utf-8") as fh:
        s = fh.read()
    m = re.search(r"SPATIAL_START_IMPLICIT\s*=\s*([^\n]+)", s)
    assert m, "SPATIAL_START_IMPLICIT 를 찾지 못했습니다"
    assert "1" in m.group(1), f"기본값이 바뀌었습니다: {m.group(1).strip()}"
