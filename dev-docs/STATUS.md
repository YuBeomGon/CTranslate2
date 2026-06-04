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

## Phase 3 — CT2 Python 바인딩 + **load-time persistent trie** (ids+bias, 토크나이저 없음)  ✅ 완료 · plan: [`p3-impl-plan.md`](p3-impl-plan.md)
> ⚠️ tokenizer compile은 여기 아님 → P4 (SSOT §0.1). trie는 WhisperWrapper에 1회 build/보관, generate 재사용(rebuild 없음).
- [x] P3-T1  C++ `build_phrase_bias_trie` + `PhraseBiasProcessor(shared_ptr<const trie>)` (shared trie)  `92a7cb89`
- [x] P3-T2  C++ `WhisperOptions.compiled_phrase_bias_trie` + generate 사용(rebuild 없음)  `a4d0f651`
- [x] P3-T3  Python ext 빌드 환경 (install-cpu + pip -e, ignore rules)  `aa023f71` / `ab4134eb`
- [x] P3-T4  `PhraseBiasPath/PhraseBias` pybind 노출 + re-export  `cab9ba65`
- [x] P3-T5  `Whisper(phrase_biases=...)` 생성자 주입(persistent) + `generate(phrase_biases=...)` override  `e91699c0`
- [x] P3-T6  whisper-tiny 통합 (생성자 persistent + per-call + empty no-op)  `15b0aadb`
- [x] **P3 완료 게이트**: C++ PhraseBiasTest 10 PASS · Python phrase bias 6 PASS · Whisper smoke 1 passed/8 skipped · 전체 C++ 회귀 196 passed/1 skipped, known-fail Gemm 3개 외 신규 실패 0

## Phase 4 — faster-whisper 연동 (tokenizer compile 포함)  ⬜ 대기 · plan: [`p4-impl-plan.md`](p4-impl-plan.md)
> 외부 사용자 진입점. 토크나이저가 여기 있음 → 키워드 compile 여기서.
- [ ] 키워드→2 path compile **pure 함수** (encode/decode 주입, special 제거, leading-space, roundtrip, step_bias)
- [ ] tokenizer correctness 테스트 (`decode(ids)==" 트랜스포머"`, 2 path)
- [ ] init 시 compile → `phrase_biases`로 CT2 generate 전달
- [ ] 실제 음성 A/B (recall + precision + insertion + CER/WER + latency)
- [x] 구현 플랜 작성 — [`p4-impl-plan.md`](p4-impl-plan.md)

---
범례: ⬜ 대기 · ⏳ 진행 중 · ✅ 완료
