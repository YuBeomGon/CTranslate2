# CLAUDE.md

이 파일은 Claude Code가 매 세션 자동으로 읽는 진입점입니다.
이 저장소는 [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2)의 **포크**이며,
STT(음성 인식) 용도로 커스텀 수정을 진행합니다.

> 상세한 설계/수정 내역은 `dev-docs/` 디렉토리를 참고하세요.
> `docs/`는 OpenNMT 공식 Sphinx 사이트 빌드용이므로 **수정하지 않습니다.**

---

## 이 포크의 목적

CTranslate2의 **Whisper 디코더에 도메인 용어 phrase bias**를 추가한다.

- **positive bias** → 도메인 정답 용어(예: "트랜스포머") recall ↑
- **negative bias** → 자주 나는 오인식(예: "트랜스퍼머") 억제 ↓

faster-whisper의 `hotwords`는 prompt hint라 next-token score를 직접 못 건드린다.
정확한 score-level 제어를 위해 기존 `LogitsProcessor` 파이프라인에 `PhraseBiasProcessor`를
**additive하게 주입**한다 (모델 포맷 변경·monkey patch 아님).

> 설계·결정·파일맵·API는 전부 [`dev-docs/SSOT.md`](dev-docs/SSOT.md)에 있다. **작업 전 SSOT를 먼저 읽어라.**

## 커스텀 문서 (Claude가 작업 전 참고)

- [`dev-docs/SSOT.md`](dev-docs/SSOT.md) — **단일 진실 공급원**. 목적·설계 결정·수정 파일맵·API·가드레일·로드맵. (가장 먼저 읽기)
- [`dev-docs/impl-plan.md`](dev-docs/impl-plan.md) — 구현 플랜 (Phase 1 block MVP, TDD). 구현 시 이걸로 진행
- [`dev-docs/testing-manual.md`](dev-docs/testing-manual.md) — 구현 후 vanilla vs fork A/B 검증 매뉴얼
- [`dev-docs/deep-research-report.md`](dev-docs/deep-research-report.md) — 전체 설계 근거/분석/벤치마크 계획 (배경 자료)
- [`dev-docs/changes.md`](dev-docs/changes.md) — upstream 대비 변경 내역 로그
- [`dev-docs/upstream-sync.md`](dev-docs/upstream-sync.md) — upstream 동기화 절차와 주의점

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
- 수정 시 `dev-docs/changes.md`에 변경 내역을 남긴다.
- 변경이 성능에 부정적 영향을 주지 않는지 확인한다 (`--log_throughput`).
