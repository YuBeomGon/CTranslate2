# CT2 KenLM Fusion Phase 1-4 Design Deltas

이 문서는 Phase 1-4 구현 중 원래 설계/계획에서 확정되거나 바뀐 지점을 기록한다.
정본 설계는 여전히 [`../ct2_kenlm_fusion_design.md`](../ct2_kenlm_fusion_design.md)이고,
구현 순서는 [`../ct2_kenlm_fusion_implementation_plan.md`](../ct2_kenlm_fusion_implementation_plan.md)를 따른다.

## 1. BeamSearch top-k buffer device

- 원래 계획: row-wise ASR top-k를 뽑은 뒤 CPU에서 KenLM score를 더한다.
- 구현 중 발견: `ops::TopK` 출력 buffer를 CPU 기본 `StorageView`로 만들면 GPU/장치 경로에서 계약이 애매해진다.
- 최종 결정: `row_topk_scores`, `row_topk_ids`는 `log_probs.device()`에 먼저 생성하고, `TopK` 후 CPU로 이동한다.
- 관련 커밋: `8203969c fix: allocate LM fusion topk buffers on logits device`

## 2. Initial history replay 위치

- 원래 계획: `DecodingOptions::lm_initial_histories`를 추가하고 prompt replay를 seed한다.
- 구현 중 확정: LM state는 opaque type이므로 BeamSearch가 직접 replay하지 않는다.
- 최종 결정: `LmFusionScorer::make_initial_states(size, initial_histories, beam_size)`가 initial history replay를 담당한다.
- 이유: KenLM state layout을 BeamSearch에 노출하지 않고, FakeScorer/KenLM scorer가 같은 인터페이스로 동작한다.

## 3. Prompt replay 범위

- 원래 논의: v1에서는 prompt replay를 막거나 제한적으로 지원하는 두 선택지가 있었다.
- 최종 결정: 제한적 지원으로 구현했다.
- 현재 정책: Whisper forwarded prompt에서 `original_id < _eot_id`인 text token만 `lm_initial_histories`에 넣고, KenLM scorer 초기화 시 replay한다.
- 남은 범위: previous-text prompt 정책과 timestamp/special token 확장 정책은 별도 검증 대상이다.

## 4. KenLM scorer source build 방식

- 원래 계획: `src/kenlm_fusion.cc`를 `WITH_KENLM=ON`일 때만 추가할 수 있다고 봤다.
- 최종 결정: `src/kenlm_fusion.cc`는 항상 컴파일한다.
- `WITH_KENLM=OFF`: factory가 명확한 runtime error를 던진다.
- `WITH_KENLM=ON`: 실제 KenLM header/library를 포함해 scorer를 생성한다.
- 이유: Whisper API wiring은 항상 `load_kenlm_bpe_scorer(...)`를 호출할 수 있어야 하고, default build에서도 링크가 깨지면 안 된다.

## 5. KenLM virtual model 사용

- 원래 계획: KenLM binary loader 구현.
- 구현 중 확정: `lm::ngram::LoadVirtual`과 `lm::base::Model`을 사용한다.
- 이유: KenLM binary가 probing/trie/quantized trie 등 어떤 타입인지 runtime에 결정될 수 있다.
- 최종 state 정책: KenLM state는 `StateSize()` byte buffer로 저장하고, `BaseFullScore`, `BeginSentenceWrite`로 갱신한다.

## 6. `KENLM_MAX_ORDER` build 계약

- 원래 계획: `KENLM_ROOT`에서 include/lib만 찾으면 된다고 봤다.
- 구현 중 발견: 로컬 KenLM 정적 라이브러리는 `KENLM_MAX_ORDER=6`으로 빌드됐지만, 이 compile definition은 외부 소비자에게 자동 전파되지 않는다.
- 최종 결정: CMake cache value `KENLM_MAX_ORDER`를 추가하고 `CT2_WITH_KENLM`과 함께 define한다.
- 주의: CT2의 `KENLM_MAX_ORDER` 값은 링크되는 KenLM library를 빌드한 값과 반드시 같아야 한다.

## 7. KenLM compression dependencies

- 원래 계획: `kenlm`, `kenlm_util` library discovery 중심.
- 구현 중 발견: 정적 KenLM 링크 시 압축 관련 symbol이 추가로 필요했다.
- 최종 결정: `WITH_KENLM=ON`일 때 `ZLIB`, `BZip2`, `LibLZMA`를 찾아 함께 링크한다.

## 8. KenLM fixture test 방식

- 원래 계획: KenLM fixture integration test 추가.
- 최종 결정: binary fixture는 repo에 넣지 않고, `CT2_KENLM_TEST_BINARY` 환경변수가 있을 때만 scorer smoke test를 실행한다.
- 이유: KenLM binary fixture의 크기/라이선스/생성 정책을 아직 확정하지 않았다.
- 로컬 검증: 임시 ARPA를 `/tmp/kenlm/build/bin/build_binary`로 변환해 `WITH_KENLM=ON` smoke test를 통과했다.

## 9. Local verification issue

- 발견: `/tmp/kenlm/build/bin/build_binary` 실행 시 Anaconda `libstdc++`가 먼저 잡혀 `GLIBCXX_3.4.32` 오류가 났다.
- 로컬 해결: `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`를 지정해 시스템 `libstdc++`를 먼저 사용했다.
- 코드 영향: 없음. 검증 환경 이슈로만 기록한다.

## 10. Shared cache 구현

- 원래 설계: path-keyed shared KenLM cache는 운영형으로 유리하지만 v1.1 후보였다.
- 최종 결정: `src/kenlm_fusion.cc::load_kenlm_bpe_scorer(...)`에 process-local shared cache를 둔다.
- Cache key: canonical path + `text_token_limit`.
- Cache value: `shared_ptr<const LmFusionScorer>`.
- 남은 hardening: 운영 artifact 변경 감지가 필요하면 file size/mtime 또는 artifact version을 cache key에 추가한다.
