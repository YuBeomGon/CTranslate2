# CT2 KenLM BPE Fusion Checklist

이 문서는 진행 상태만 관리한다.
정책은 [ct2_kenlm_fusion_design.md](ct2_kenlm_fusion_design.md), 구현 세부는 [ct2_kenlm_fusion_implementation_plan.md](ct2_kenlm_fusion_implementation_plan.md)를 따른다.

## 1. Documentation

- [x] Reference 문서 3개를 `dev-docs/reference/`로 모음
- [x] Design SSOT 문서 작성
- [x] Implementation plan 문서 작성
- [x] Checklist 문서 작성
- [x] `AGENTS.md`에 SSOT 링크와 문서 규칙 추가
- [ ] 구현 중 design 결정 변경 시 design 문서 갱신
- [ ] 구현 완료 후 benchmark/result 문서 추가

## 2. Decoder Mapping

- [x] `Decoder`에 `_effective_output_size` 추가
- [x] `Decoder::effective_output_size()` 추가
- [x] `Decoder::is_padding_output_id()` 추가
- [x] `update_output_layer()`에서 padding 전 effective size 저장
- [x] output layer reset 경로에서 effective size 정리
- [x] `tests/model_test.cc`에 padding accessor test 추가

## 3. Internal Fusion API

- [x] `include/ctranslate2/lm_fusion.h` 추가
- [x] `LmFusionOptions` 정의
- [x] `LmStateBatch` interface 정의
- [x] `LmFusionScorer` interface 정의
- [x] `LmFusionScorer::gather()` 정의
- [x] `LmFusionScorer::keep_batches()` 정의
- [x] `DecodingOptions`에 `lm_fusion` 추가
- [x] `DecodingOptions`에 `lm_fusion_scorer` 추가
- [x] `DecodingOptions`에 `lm_initial_histories` 추가
- [x] `BeamSearch` constructor에 fusion options/scorer 주입
- [x] `make_search_strategy()` wiring

## 4. Validation

- [x] fusion enabled helper 추가
- [ ] `alpha > 0` validation
- [x] `asr_topk > 0` validation
- [x] scorer null validation
- [x] `beam_size > 1` validation
- [x] deterministic sampler path validation
- [x] `return_alternatives == false` validation
- [x] `asr_topk <= vocabulary_size` validation
- [ ] unsupported mode tests 추가

## 5. BeamSearch Fusion

- [ ] fusion enabled 시 `expand_after_first_step` 비활성화
- [ ] LM state 초기화
- [ ] prompt history replay hook 추가
- [ ] row-wise ASR top-k 추출
- [ ] output id padding skip
- [ ] output id to original id 변환
- [ ] text token LM score 적용
- [ ] special/timestamp skip state copy
- [ ] fused score 계산
- [ ] batch top `num_candidates` 선택
- [ ] `topk_ids`는 output token id로 유지
- [ ] `gather_indices`는 beam origin으로 유지
- [ ] candidate 부족 시 explicit error
- [ ] hard-prefix forced step whole-step 우회 처리
- [ ] prefix update 후 최종 token 기준 candidate LM state 생성
- [ ] active beam 선택 후 `LmFusionScorer::gather()`로 LM state gather
- [ ] finished batch prune 후 `LmFusionScorer::keep_batches()`로 LM state prune

## 6. Whisper API Wiring

- [ ] `WhisperOptions::lm_fusion_model_path` 추가
- [ ] `WhisperOptions::lm_fusion_alpha` 추가
- [ ] `WhisperOptions::lm_fusion_asr_topk` 추가
- [ ] `WhisperOptions::lm_fusion_debug` 추가
- [ ] `WhisperReplica::generate()`에서 `text_token_limit = _eot_id` 전달
- [ ] prompt replay용 initial history 전달
- [ ] initial history는 `original_id < _eot_id` text token으로 제한
- [ ] Python `WhisperWrapper::generate()` signature 확장
- [ ] pybind keyword list 확장
- [ ] Python docstring에 fused score 의미 명시

## 7. KenLM Build and Scorer

- [ ] `WITH_KENLM` CMake option 추가
- [ ] `KENLM_ROOT` cache path 추가
- [ ] KenLM include discovery 추가
- [ ] KenLM library discovery 추가
- [ ] `CT2_WITH_KENLM` compile definition 추가
- [ ] `WITH_KENLM=OFF` 요청 error 구현
- [ ] KenLM binary loader 구현
- [ ] token id to `t<ID>` word index table 구현
- [ ] log10 to natural log 변환 구현
- [ ] text token state advance 구현
- [ ] skip token state copy 구현
- [ ] path-keyed shared cache 도입 여부 결정

## 8. Tests

- [x] Decoder padding id accessor test
- [ ] fusion-off baseline parity test
- [ ] top-k 내부 LM winner test
- [ ] top-k 밖 no-rescue test
- [ ] special/timestamp skip test
- [ ] beam reorder state alignment test
- [ ] finished batch prune state alignment test
- [ ] hard-prefix state sync test
- [ ] fused EOS/final score test
- [ ] `WITH_KENLM=OFF` error smoke
- [ ] KenLM fixture integration test
- [ ] Python API smoke test

## 9. Benchmark and Evaluation

- [ ] HF POC 최신 metric 재측정
- [ ] baseline CT2 latency 측정
- [ ] fusion topk=20 latency/quality 측정
- [ ] fusion topk=50 latency/quality 측정
- [ ] fusion topk=100 latency/quality 측정
- [ ] alpha sweep 측정
- [ ] p50/p95 latency 기록
- [ ] KenLM query count 기록
- [ ] CPU fusion time 기록
- [ ] domain term recall/precision 기록
- [ ] CER/WER 기록

## 10. Release Readiness

- [ ] `WITH_KENLM=OFF` default build 확인
- [ ] `WITH_KENLM=ON` build 확인
- [ ] C++ tests 통과
- [ ] Python smoke 통과
- [ ] benchmark 결과 문서화
- [ ] KenLM license 검토
- [ ] packaging 정책 결정
