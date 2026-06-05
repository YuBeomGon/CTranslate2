# CLAUDE.md

이 파일은 Claude Code가 매 세션 자동으로 읽는 진입점입니다.
이 저장소는 [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2)의 **포크**이며,
STT(음성 인식) 용도로 커스텀 수정을 진행합니다.

> 상세한 설계/수정 내역은 `dev-docs/` 디렉토리를 참고하세요.
> `docs/`는 OpenNMT 공식 Sphinx 사이트 빌드용이므로 **수정하지 않습니다.**

---

## 이 포크의 목적

CTranslate2의 **Whisper 디코더에 도메인 용어 signed phrase bias**를 추가한다.
문장 아무 위치에서 도메인 phrase가 나오려 할 때 **다음 token logit에 양수/음수 bias**를 더해
도메인 용어 recall을 올리거나 오인식 후보를 soft suppress한다.
hard block/suppress는 보류. 기존 `LogitsProcessor` 파이프라인에 `PhraseBiasProcessor`를 additive 주입.

> **모든 결정·범위·금지사항·테스트·구현 참조는 [`dev-docs/SSOT.md`](dev-docs/SSOT.md) 하나에 통합되어 있다. 작업 전 SSOT를 먼저, 끝까지 읽어라.**

## 커스텀 문서

- [`dev-docs/SSOT.md`](dev-docs/SSOT.md) — **단일 진실 공급원** (통합 문서: 결정·로드맵·code map·금지사항·테스트·exemplar·검증)
- `dev-docs/archive/` — 배경 리서치 + 이전 분절 문서 (참고용)

---

## 빌드 / 테스트 명령어

### C++ 테스트
```bash
cmake -DBUILD_TESTS=ON ...        # 테스트 활성화
./tests/ctranslate2_test ../tests/data
```

### Python 테스트
```bash
cd python
pip install -r tests/requirements.txt
pytest tests/
```

### 코드 스타일 (Python)
```bash
black .
isort .
flake8 .
```

### 성능 측정
- `--log_throughput`: 초당 생성 토큰 수 (런 간 비교용, 높을수록 좋음)
- `--log_profiling`: 함수별 실행 프로파일

---

## 코드베이스 핵심 구조

성능이 매우 중요한 저수준 코드베이스입니다. 포인터/메모리 할당 실수는 디버깅이 오래 걸리니 주의.

추상화 레벨 (낮음 → 높음):
- **kernels** — 저수준 연산 함수 (예: CUDA Softmax)
- **primitives** — 기본 벡터/행렬 처리 (`include/ctranslate2/primitives.h`)
- **ops** — 신경망 연산 (Softmax, Gemm 등) → `src/ops/`, `include/ctranslate2/ops/`
- **layers** — 상태 있는 레이어 (Dense, LayerNorm) → `src/layers/`
- **models** — 레이어+가중치 묶음 (Transformer) → `src/models/`
- **replicas / replica pool** — 실행 인스턴스, 스레드 풀

핵심 데이터 구조:
- `StorageView` (`include/ctranslate2/storage_view.h`) — row-major 텐서 래퍼.
  타입/디바이스는 런타임 결정. 재할당 최소화(캐싱 할당자) 설계.

새 Op 추가 시 보통 필요한 파일:
```
include/ctranslate2/ops/my_op.h   # 인터페이스 (컴파일 플래그 금지)
src/ops/my_op.cc                  # 입력 검증 + device/type 디스패치
src/ops/my_op_cpu.cc              # CPU 구현
src/ops/my_op_gpu.cu              # CUDA 구현
```

---

## 작업 규칙

- `docs/` (공식 Sphinx)는 건드리지 않는다. 내 문서는 전부 `dev-docs/`.
- 코드 동작, API, 기본값, 설정 스키마가 바뀌면 반드시 `dev-docs/SSOT.md`와 관련 `dev-docs/` 문서를 같은 변경에서 갱신한다. 코드와 문서가 불일치하면 작업 미완료로 본다.
- 수정 시 `dev-docs/changes.md`에 변경 내역을 남긴다.
- 변경이 성능에 부정적 영향을 주지 않는지 확인한다 (`--log_throughput`).
