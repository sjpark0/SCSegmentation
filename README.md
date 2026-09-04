# SCSegmentation

다중시점 비디오 객체 분할 실험 코드베이스.

- `SCSam2/`, `SCSam3/` — SAM 2 / SAM 3 기반 다중시점 세그멘테이션
- `Data/MVSeg/` — MVSeg 벤치마크와 J&F 평가 (`eval_jf.py`, `report_jf.py`)
- `figures/` — 논문용 그림 생성 스크립트

## Docker 이미지 빌드

SAM 3 체크포인트는 gated 저장소(`facebook/sam3`)에 있습니다. 두 가지 방법이 있고,
**어느 쪽이든 토큰이 이미지 레이어에 남지 않습니다.**

### 방법 1 — 로컬 웨이트 (권장, 토큰 불필요)

`SCSam3/hf_cache/`에 웨이트가 있으면 그대로 복사해 씁니다.

```bash
docker build -t scsam3 SCSam3/
```

웨이트가 없는 머신에서는 **한 번만** 내려받으면 됩니다:

```bash
pip install huggingface_hub
cd SCSam3
HF_HOME=./hf_cache hf download facebook/sam3 --token <토큰>
```

이 다운로드에만 토큰이 필요하고, 빌드에는 들어가지 않습니다.
`hf_cache/`의 내용은 gitignore 대상입니다(6.5 GB). 디렉터리 자체는 `.gitkeep`으로 유지됩니다.

### 방법 2 — 빌드 중 다운로드 (신규 클론 등)

웨이트가 없으면 BuildKit secret으로 토큰을 전달합니다. 토큰은 그 한 명령에만 마운트되고
레이어에 기록되지 않습니다.

```bash
export HF_TOKEN=...            # 셸에서만, 파일에 쓰지 말 것
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t scsam3 SCSam3/
```

둘 다 없으면 **빌드가 즉시 실패합니다** — 첫 실행 때가 아니라 빌드 시점에.

### 빌드 컨텍스트

`SCSam3/.dockerignore`가 컨텍스트를 `sam3/`와 `hf_cache/`로 제한합니다.
이 파일이 없으면 데모 출력 PNG까지 포함해 **84 GB**가 전송됩니다(현재 6.6 GB).
