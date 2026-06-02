# 진행 상태 (Phrase Bias)

> task 완료(테스트 통과 + 리뷰) 시마다 `- [ ]` → `- [x]` 체크하고 commit SHA 기록 후 커밋.
> 스펙 = [`SSOT.md`](SSOT.md) · P1 상세 = [`p1-impl-plan.md`](p1-impl-plan.md)

## Phase 1 — CPU positive bias + reverse trie  ✅ 완료
- [x] (사전) feature 브랜치 + 데이터 모델 `whisper.h`            `34e52dd`
- [x] P1-T1  `indexed_add` CPU primitive (+ test)                  `975e26f8`
- [x] P1-T2  `PhraseBiasEntry` + reverse trie (build/lookup)       `16ad7611`
- [x] P1-T3  `PhraseBiasProcessor` CPU apply (합산+clamp, synthetic logits)  `155ace85`
- [x] P1-T4  generate 주입 + empty no-op 회귀                      `e849bea6`
- [x] **P1 완료 게이트**: 전체 C++ 테스트 통과 (baseline Gemm 3개 제외) — 194 passed

## Phase 2 — GPU `indexed_add` (parity)  ✅ 완료  · plan: [`p2-impl-plan.md`](p2-impl-plan.md)
- [x] P2-T1  CUDA 빌드 디렉터리 셋업 (`build-cuda/` Auto + `build-cuda-portable/` Common)
- [x] P2-T2  CUDA `indexed_add` 커널 + primitive parity (device-param fp32)  `38a5b4c0`
- [x] P2-T3  `PhraseBiasProcessor` CPU/GPU parity (CPU fp32 + CUDA fp32/fp16/bf16, batch>1, overlap)  `424a7450`
- [x] **P2 완료 게이트**: 1차 게이트 17 PASS · 전체 회귀 CUDA 363 passed (known-fail Gemm 3개 외 신규 0) · portable Common 멀티-arch(sm_53~86+PTX) 컴파일 OK

## Phase 3 — CT2 Python 바인딩만 (ids+bias, **토크나이저 없음**)  ⬜ 대기
> ⚠️ tokenizer compile은 여기 아님 → P4. CT2엔 토크나이저 없음 (SSOT §0.1).
- [ ] `PhraseBiasMode/Path/Bias` pybind `py::class_` 노출 (`python/cpp/whisper.cc`)
- [ ] `generate(phrase_biases=...)` kwarg (suppress_tokens 패턴) → `WhisperOptions.phrase_biases`
- [ ] 바인딩 왕복 테스트 (py list/dict → C++ → 동작) + empty no-op
- [ ] whisper-tiny generate 통합 테스트 (ids+bias로 decode result 변경)
- [ ] (plan 작성 필요)

## Phase 4 — faster-whisper 연동 (tokenizer compile 포함)  ⬜ 대기
> 외부 사용자 진입점. 토크나이저가 여기 있음 → 키워드 compile 여기서.
- [ ] 키워드→2 path compile **pure 함수** (encode/decode 주입, special 제거, leading-space, roundtrip, step_bias)
- [ ] tokenizer correctness 테스트 (`decode(ids)==" 트랜스포머"`, 2 path)
- [ ] init 시 compile → `phrase_biases`로 CT2 generate 전달
- [ ] 실제 음성 A/B (recall + precision + insertion + CER/WER + latency)
- [ ] (plan 작성 필요 — faster-whisper repo)

---
범례: ⬜ 대기 · ⏳ 진행 중 · ✅ 완료
