# SCSegmentation

다중시점 비디오 객체 분할 실험 코드베이스.

- `SCSam2/`, `SCSam3/` — SAM 2 / SAM 3 기반 다중시점 세그멘테이션
- `Data/MVSeg/` — MVSeg 벤치마크와 J&F 평가 (`eval_jf.py`, `report_jf.py`)
- `figures/` — 논문용 그림 생성 스크립트

## HuggingFace 인증

체크포인트는 gated 저장소(`facebook/sam3`)에서 받습니다. 토큰을 저장소나
Dockerfile에 넣지 마십시오. 대신 빌드 시 BuildKit secret으로 전달합니다:

```bash
export HF_TOKEN=...            # 셸에서만, 파일에 쓰지 말 것
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t scsam3 SCSam3/
```
