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

## Phase 2 — GPU `indexed_add` (parity)  ⬜ 대기
- [ ] CUDA `indexed_add` (`indexed_fill` 패턴, unique index)
- [ ] CPU/GPU parity 테스트 (fp32 타이트 / fp16 tolerance / beam 1·5 / batch>1)
- [ ] (plan 작성 필요)

## Phase 3 — Python binding + tokenizer compile  ⬜ 대기
- [ ] `phrase_biases` / `phrase_bias_config` kwarg
- [ ] tokenizer compile (special 제거, **leading-space 유지**, space/no-space 두 path)
- [ ] tokenizer correctness 테스트 (`decode(ids)==" 트랜스포머"`)
- [ ] (plan 작성 필요)

## Phase 4 — faster-whisper 연동  ⬜ 대기
- [ ] 도메인 모델 init 시 vocab baking
- [ ] 실제 음성 A/B (recall + precision + insertion + CER/WER + latency)
- [ ] (plan 작성 필요)

---
범례: ⬜ 대기 · ⏳ 진행 중 · ✅ 완료
