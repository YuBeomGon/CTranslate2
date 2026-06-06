# CT2 KenLM BPE Fusion Implementation Plan

> 목적: HuggingFace POC에서 검증한 BPE token-id KenLM 1-pass shallow fusion을 CTranslate2 Whisper 디코더 내부로 이식한다.

이 문서는 `../CTranslate2` 로컬 저장소의 현재 코드 구조를 기준으로 작성했다.

```text
CT2 repo   : ../CTranslate2
CT2 branch : feature/kenlm-bpe-fusion
CT2 remote : https://github.com/YuBeomGon/CTranslate2
```

## 1. 목표

HF POC의 점수 결합을 CT2 beam search 후보 선택 경로에 넣는다.

```text
score(candidate) = CT2_ASR_logprob + beam_cumulative_score + alpha * KenLM_BPE_logprob
```

KenLM은 Whisper BPE token id를 pseudo-word로 변환한 corpus로 학습한다.

```text
Whisper token id 1234 -> KenLM word "t1234"
```

1차 구현은 `top-k only` fusion이다. 즉, ASR top-k 후보 안에서만 KenLM 점수를 더해 재정렬한다.

### 1.1 구현 전 선행 확인

- HF POC 결과는 `mode=topk`, `asr_topk=50`, `beam=5`, `n_best=5` 기준이다.
- 현재 HF `topk` 구현은 ASR top-k 후보에만 LM 점수를 더하는 근사이며, full-vocab shallow fusion의 정확한 등가가 아니다.
- domain-term metric은 longest-match/공백 무시 정책으로 변경됐으므로, 기존 결과 문서의 recall/precision 숫자는 최신 코드로 재측정해야 한다.
- `max_audio_seconds=30` 초과 샘플은 평가에서 제외하거나 segment 단위 reference와 맞춰야 한다. 현재 POC 코드는 제외 정책을 적용한다.
- CT2 이식 착수 게이트는 최신 metric과 동일한 audio 길이 정책으로 다시 산출한 HF 결과를 기준으로 한다.

## 2. 비목표

- union/domain-successor 후보 확장은 1차 범위에서 제외한다.
- Python per-step callback 방식은 사용하지 않는다.
- random sampling fusion은 1차 범위에서 제외한다.
- CT2 model conversion 또는 model binary format은 바꾸지 않는다.
- KenLM 학습은 CT2 안에서 하지 않는다. 학습 산출물인 `.binary`만 로드한다.

## 3. 현재 CT2 코드 경로

### 3.1 Whisper 진입점

`../CTranslate2/include/ctranslate2/models/whisper.h`

- `models::WhisperOptions`가 Whisper generate 옵션을 정의한다.
- 여기에 LM fusion 옵션을 추가한다.

`../CTranslate2/python/cpp/whisper.cc`

- `WhisperWrapper::generate()`가 Python keyword를 받아 `WhisperOptions`를 채운다.
- pybind `.def("generate", ...)`에 Python API keyword를 추가한다.

`../CTranslate2/src/models/whisper.cc`

- `WhisperReplica::generate()`가 `DecodingOptions`를 구성한다.
- 현재 흐름:

```text
Whisper options
  -> prompt/state 준비
  -> DecodingOptions 구성
  -> timestamp/no-speech LogitsProcessor 추가
  -> decode(*_decoder, state, start_tokens, {_eot_id}, decoding_options)
```

### 3.2 Beam search 핵심 경로

`../CTranslate2/src/decoding.cc`

`BeamSearch::search()`가 실제 beam loop를 수행한다.

현재 핵심 순서:

```text
decoder(...)                         # step logits 생성
LogitsProcessor 적용                 # suppress/timestamp/no-repeat 등
disable_tokens.apply()
LogSoftMax
current beam score broadcast add
flatten [batch, beam, vocab]
sampler(..., num_candidates)          # top candidate 선택
unflatten_ids(...)                    # token id + beam origin 계산
EOS/finished 처리
active_beams 계산
decoder.update_state(...)             # CT2 decoder KV/state reorder
```

LM fusion은 `LogSoftMax + beam score add` 이후, `sampler()` 후보 선택 전에 들어가야 한다.

## 4. 핵심 설계 결정

### 4.1 `LogitsProcessor`가 아니라 `BeamSearch` 내부에 구현한다

CT2의 `LogitsProcessor`는 logits와 alive sequence를 볼 수 있지만, beam별 외부 LM state를 소유하거나 candidate 선택 이후 state를 reorder하기 어렵다.

KenLM fusion은 다음 상태가 필요하다.

```text
active beam state
  -> candidate token별 next KenLM state
  -> beam selection 결과에 따라 next state reorder
```

따라서 1차 구현은 `BeamSearch::search()` 내부에서 처리한다.

### 4.2 Phase 1은 ASR top-k strict 후보만 사용한다

HF POC의 `topk` 방식은 top-k 토큰에만 보통 음수인 LM logprob를 더하고 non-top-k는 그대로 두는 해석 문제가 있다. CT2 이식에서는 이 문제를 피하기 위해 후보 집합 자체를 ASR top-k로 제한한다.

```text
for each beam:
  ASR top lm_fusion_asr_topk 후보 추출
  각 후보에 alpha * LM score 가산
batch 단위로 fused score top num_candidates 선택
```

ASR top-k 밖 token은 1차 구현에서 rescue하지 않는다. 첫-token rescue는 추후 union/domain-successor 확장 단계에서 다룬다.

### 4.3 1차 지원 범위

지원:

- Whisper beam search
- `sampling_topk == 1`
- `sampling_temperature == 1`
- `return_alternatives == false`

거부:

- random sampling + LM fusion
- alternatives mode + LM fusion
- union expansion
- Python callback 기반 token streaming fusion

## 5. Public API 계획

### 5.1 C++ `WhisperOptions`

파일:

```text
../CTranslate2/include/ctranslate2/models/whisper.h
```

추가 필드:

```cpp
// Path to a KenLM binary trained on Whisper BPE pseudo words t<ID>.
std::string lm_fusion_model_path;

// Fusion weight. Disabled when <= 0 or when lm_fusion_model_path is empty.
float lm_fusion_alpha = 0;

// Number of ASR candidates per beam to rescore with KenLM.
size_t lm_fusion_asr_topk = 50;

// Optional debug logging/counters.
bool lm_fusion_debug = false;
```

기본값에서는 기존 CT2 동작이 완전히 유지되어야 한다.

### 5.2 Python API

파일:

```text
../CTranslate2/python/cpp/whisper.cc
```

`Whisper.generate()` keyword 추가:

```python
lm_fusion_model_path: Optional[str] = None
lm_fusion_alpha: float = 0
lm_fusion_asr_topk: int = 50
lm_fusion_debug: bool = False
```

예상 사용:

```python
result = whisper.generate(
    features,
    prompts,
    beam_size=5,
    num_hypotheses=5,
    return_scores=True,
    lm_fusion_model_path="data/lm_honest_5g.binary",
    lm_fusion_alpha=0.1,
    lm_fusion_asr_topk=50,
)
```

LM fusion이 켜진 경우 `scores`는 ASR-only score가 아니라 fused score로 정의한다.

## 6. Internal API 계획

### 6.1 `DecodingOptions` 확장

파일:

```text
../CTranslate2/include/ctranslate2/decoding.h
```

추가 구조:

```cpp
struct LmFusionOptions {
  float alpha = 0;
  size_t asr_topk = 50;
  size_t text_token_limit = 0;  // Whisper에서는 _eot_id.
  bool debug = false;
};
```

`DecodingOptions`에 추가:

```cpp
std::shared_ptr<const LmFusionScorer> lm_fusion_scorer;
LmFusionOptions lm_fusion;
```

`WhisperReplica::generate()`에서 다음 조건일 때 scorer를 구성한다.

```text
!options.lm_fusion_model_path.empty()
&& options.lm_fusion_alpha > 0
```

그리고 다음 값을 전달한다.

```text
decoding_options.lm_fusion_scorer = scorer
decoding_options.lm_fusion.alpha = options.lm_fusion_alpha
decoding_options.lm_fusion.asr_topk = options.lm_fusion_asr_topk
decoding_options.lm_fusion.text_token_limit = _eot_id
decoding_options.lm_fusion.debug = options.lm_fusion_debug
```

Whisper vocabulary layout상 `_eot_id` 이전이 일반 text token이고, `_eot_id` 이후는 special/language/task/timestamp 영역이다.

## 7. KenLM 의존성 계획

### 7.1 CMake option

파일:

```text
../CTranslate2/CMakeLists.txt
```

추가:

```cmake
option(WITH_KENLM "Compile KenLM shallow fusion support" OFF)
```

`WITH_KENLM=ON`일 때만 KenLM include/lib를 찾고 링크한다.

예상 형태:

```cmake
set(KENLM_ROOT "" CACHE PATH "Path to KenLM install/build root")
find_path(KENLM_INCLUDE_DIR lm/model.hh HINTS ${KENLM_ROOT}/include ${KENLM_ROOT})
find_library(KENLM_LIBRARY kenlm HINTS ${KENLM_ROOT}/lib ${KENLM_ROOT}/build/lib)
find_library(KENLM_UTIL_LIBRARY kenlm_util HINTS ${KENLM_ROOT}/lib ${KENLM_ROOT}/build/lib)

add_definitions(-DCT2_WITH_KENLM)
list(APPEND PRIVATE_INCLUDE_DIRECTORIES ${KENLM_INCLUDE_DIR})
list(APPEND LIBRARIES ${KENLM_LIBRARY} ${KENLM_UTIL_LIBRARY})
```

실제 library 이름은 KenLM build 산출물 기준으로 구현 전에 확인한다.

### 7.2 `WITH_KENLM=OFF` 동작

기본 build는 기존과 동일해야 한다.

`WITH_KENLM=OFF`인데 `lm_fusion_model_path`가 들어오면 명확히 실패한다.

```text
KenLM fusion requires CTranslate2 built with WITH_KENLM=ON
```

## 8. Scorer 설계

추가 파일:

```text
../CTranslate2/include/ctranslate2/kenlm_fusion.h
../CTranslate2/src/kenlm_fusion.cc
```

인터페이스 초안:

```cpp
class LmFusionScorer {
public:
  virtual ~LmFusionScorer() = default;

  virtual bool is_text_token(size_t original_token_id) const = 0;

  virtual KenlmState start_state() const = 0;

  virtual float score(const KenlmState& state,
                      size_t original_token_id,
                      KenlmState* out_state) const = 0;
};
```

KenLM 구현:

```cpp
class KenlmBpeScorer final : public LmFusionScorer {
public:
  KenlmBpeScorer(const std::string& binary_path,
                 size_t text_token_limit);
};
```

구현 규칙:

- original Whisper token id `i`를 `"t" + std::to_string(i)`로 매핑한다.
- `text_token_limit` 크기의 `std::vector<lm::WordIndex>`를 미리 만든다.
- `original_token_id >= text_token_limit`이면 LM score는 `0`, state는 그대로 유지한다.
- KenLM 반환 `FullScore.prob`는 log10이므로 natural log로 변환한다.

```cpp
lm_score_ln = full_score.prob * std::log(10.0f);
```

## 9. Beam별 KenLM State 설계

### 9.1 State 배열

`BeamSearch::search()` 내부에서 fusion이 켜졌을 때만 유지한다.

```cpp
std::vector<KenlmState> lm_states;
std::vector<KenlmState> candidate_lm_states;
```

`lm_states` 정렬:

```text
shape: cur_batch_size * beam_size
index: batch_index * beam_size + beam_index
```

`candidate_lm_states` 정렬:

```text
shape: cur_batch_size * num_candidates
index: batch_index * num_candidates + candidate_index
```

### 9.2 초기화

fusion enabled일 때:

1. 입력 batch마다 KenLM BeginSentence state를 만든다.
2. decode prefix에 포함된 text token을 replay한다.
3. beam 확장 시 beam size만큼 replicate한다.

1차 구현에서는 CPU-only 첫 step 최적화인 `expand_after_first_step`을 fusion enabled일 때 끈다.

```cpp
const bool expand_after_first_step =
  !lm_fusion_enabled
  && device == Device::CPU
  && num_candidates <= vocabulary_size;
```

이렇게 하면 CPU/GPU 모두에서 `cur_batch_size * beam_size` 정렬이 동일해져 LM state alignment가 단순해진다.

### 9.3 skip token 정책

skip token:

```text
LM additive score = 0
candidate next state = previous LM state
```

Whisper에서는 `_eot_id` 이상 token을 모두 skip한다.

포함되는 token:

- `<|endoftext|>`
- `<|startoftranscript|>`
- language tokens
- task tokens
- `<|nospeech|>`
- `<|notimestamps|>`
- timestamp tokens

## 10. CT2 output id 변환 주의

`WhisperReplica::generate()`는 `_decoder->update_output_layer(...)`를 호출한다. 이 경우 decode 내부 candidate id가 original vocab id가 아닐 수 있다.

KenLM은 original Whisper token id 기준으로 학습되어 있으므로 scoring 전에 변환해야 한다.

```cpp
size_t original_id = output_id;
if (decoder.output_layer_is_updated()) {
  if (!decoder.is_in_output(output_id)) {
    // real vocab token이 아니면 score/선택 대상에서 제외하거나 skip 처리
  }
  original_id = decoder.to_original_word_id(output_id);
}
```

CT2 beam search의 `topk_ids`에는 기존대로 output id를 유지한다. KenLM scoring에만 original id를 사용한다.

## 11. Candidate Selection 알고리즘

현재 CT2:

```cpp
ops::LogSoftMax()(logits);
add_depth_broadcast(topk_scores, log_probs);
log_probs.reshape({cur_batch_size, -1});
sampler(log_probs, topk_ids, topk_scores, num_candidates);
StorageView gather_indices = unflatten_ids(...);
```

LM fusion enabled:

```text
1. log_probs shape는 [cur_batch_size * beam_size, vocab]
2. CT2 기존 방식대로 current beam score를 더한다.
3. 각 active beam row에서 ASR top lm_fusion_asr_topk token을 뽑는다.
4. 각 token에 대해:
   - output id -> original Whisper id 변환
   - text token이면 KenLM FullScore
   - special/timestamp이면 LM score 0
   - fused_score = asr_score + alpha * lm_score
   - candidate next state 저장
5. batch 단위로 beam_size * asr_topk 후보를 fused_score 기준 정렬한다.
6. 상위 num_candidates를 topk_ids/topk_scores에 기록한다.
   - topk_ids는 flat id: beam_id * vocabulary_size + output_token_id
   - topk_scores는 fused cumulative score
7. 기존 unflatten_ids(), EOS 처리, active_beams 처리로 이어간다.
```

이 방식은 기존 CT2 구조인 `topk_ids`, `topk_scores`, `gather_indices`, `active_beams`, `decoder.update_state()`를 최대한 유지한다.

## 12. LM State Reorder 알고리즘

기존 CT2는 active beam을 이렇게 고른다.

```cpp
active_beams.at<int32_t>(i * _beam_size + k)
  = i * num_candidates + next_beam_id;
```

이후 token/score/decoder state를 gather한다.

LM state도 같은 선택 결과를 적용한다.

```text
next_lm_states[i * beam_size + k]
  = candidate_lm_states[i * num_candidates + next_beam_id]
```

일부 batch가 종료되어 `non_finished_index`로 줄어들 때도 같은 순서로 LM state를 줄인다.

```text
lm_states = live batches only, using non_finished_index
```

이 부분은 CT2 decoder state reorder와 반드시 같은 논리를 따라야 한다.

## 13. 기존 기능과의 상호작용

### suppress/timestamp rules

순서는 유지한다.

```text
LogitsProcessor
disable_tokens.apply()
LogSoftMax
LM fusion candidate scoring
```

따라서 LM은 suppress된 token을 되살리지 않는다.

### length penalty

변경하지 않는다.

`topk_scores`가 fused cumulative score가 되고, 기존 `finalize_result()`가 length penalty를 적용한다.

### return_scores

LM fusion enabled일 때 반환 score는 fused score다. 문서와 Python docstring에 명시한다.

### return_logits_vocab

1차 구현에서는 기존 의미를 유지한다. 즉, returned logits는 LM fusion 전 CT2 model logits/log-probs다. fused candidate score가 필요하면 별도 debug 출력으로 추가한다.

### no-speech probability

LM fusion을 적용하지 않는다. no-speech probability는 기존 `GetNoSpeechProbs`와 `get_no_speech_probs_from_logits()` 흐름을 유지한다.

## 14. Validation 규칙

`validate_decoding_options()`에 추가한다.

```text
if lm_fusion enabled:
  alpha > 0
  asr_topk > 0
  scorer != nullptr
  sampling_topk == 1
  sampling_temperature == 1
  return_alternatives == false
  asr_topk <= vocabulary_size
```

지원하지 않는 조합은 조용히 무시하지 말고 예외를 던진다.

## 15. 테스트 계획

### 15.1 KenLM 없이 가능한 unit test

fake scorer를 만들어 beam logic을 테스트한다.

권장 테스트:

1. **Top-k 안의 LM 선호 token이 winner로 올라온다**
   - ASR rank 1은 아니지만 ASR top-k 안에 있는 token을 fake scorer가 선호.
   - fused 결과가 해당 token을 선택.

2. **ASR top-k 밖 token은 Phase 1에서 rescue되지 않는다**
   - fake scorer가 매우 강하게 선호해도 ASR top-k 밖이면 선택되지 않음.

3. **special/timestamp skip은 LM state를 advance하지 않는다**
   - `original_id >= text_token_limit` 후보에서 state가 그대로 유지됨.

4. **beam reorder 후 LM state alignment가 유지된다**
   - step 1에서 secondary beam 후보가 선택됨.
   - step 2의 fake scorer가 이전 token history에 의존.
   - 결과로 state mismatch를 잡아낸다.

5. **finished hypothesis score는 fused score를 쓴다**
   - EOS 후보가 선택될 때 `result.scores`가 fused cumulative score와 일치.

6. **unsupported mode validation**
   - sampling + LM fusion 실패.
   - `asr_topk == 0` 실패.
   - scorer 없음 + alpha > 0 실패.

### 15.2 KenLM integration test

작은 corpus:

```text
t7 t8
t7 t8
t3 t4
```

KenLM binary가 test fixture로 있을 때만 실행하고, 없으면 skip한다.

검증:

- log10 -> ln 변환
- text token state advance
- special token score 0 + state 유지

### 15.3 HF parity check

전체 transcript parity를 바로 요구하지 않는다. 대신 동일한 candidate fixture에서 Python reference와 CT2 candidate reorder 결과를 비교한다.

비교 기준:

- 같은 top-k 후보
- 같은 KenLM binary
- 같은 alpha
- fused candidate ranking 동일

## 16. Benchmark 계획

동일 모델/동일 audio subset에서 비교한다.

```text
baseline CT2
CT2 + LM fusion asr_topk=20
CT2 + LM fusion asr_topk=50
```

측정:

- p50 latency
- p95 latency
- throughput
- average decode steps
- KenLM query count
- CPU time in fusion section
- GPU utilization

예상 query count:

```text
decode_steps * beam_size * asr_topk
```

예:

```text
80 steps * beam 5 * topk 50 = 20,000 KenLM score calls / utterance
```

## 17. 구현 Task

### Task 1. CMake KenLM flag

파일:

```text
../CTranslate2/CMakeLists.txt
```

작업:

- `WITH_KENLM` option 추가
- KenLM include/lib discovery 추가
- `CT2_WITH_KENLM` compile definition 추가
- 기본 build에서는 기존 동작 유지

### Task 2. Whisper public API 추가

파일:

```text
../CTranslate2/include/ctranslate2/models/whisper.h
../CTranslate2/python/cpp/whisper.cc
```

작업:

- `WhisperOptions`에 LM fusion 필드 추가
- Python keyword 추가
- docstring에 fused score 의미 명시

### Task 3. KenLM scorer 추가

파일:

```text
../CTranslate2/include/ctranslate2/kenlm_fusion.h
../CTranslate2/src/kenlm_fusion.cc
../CTranslate2/CMakeLists.txt
```

작업:

- scorer interface 추가
- `KenlmBpeScorer` 구현
- `t<ID>` mapping
- `FullScore` log10 -> ln 변환
- special/timestamp skip

### Task 4. DecodingOptions 확장 및 validation

파일:

```text
../CTranslate2/include/ctranslate2/decoding.h
../CTranslate2/src/decoding.cc
```

작업:

- `LmFusionOptions` 추가
- scorer pointer 추가
- unsupported mode validation 추가

### Task 5. BeamSearch top-k fusion 구현

파일:

```text
../CTranslate2/src/decoding.cc
```

작업:

- fusion enabled 시 `expand_after_first_step` 비활성화
- beam별 KenLM state 초기화/replicate
- ASR top-k 후보 추출
- 후보별 KenLM score 가산
- fused candidate top `num_candidates` 선택
- candidate LM state 저장
- active beam 선택 후 LM state reorder
- finished batch 제거 시 LM state도 동일하게 제거

### Task 6. Whisper -> DecodingOptions wiring

파일:

```text
../CTranslate2/src/models/whisper.cc
```

작업:

- `lm_fusion_model_path`가 있으면 scorer 생성
- `text_token_limit = _eot_id`
- `DecodingOptions`에 scorer/options 전달

### Task 7. 테스트 및 벤치

파일:

```text
../CTranslate2/tests/decoding_test.cc
../CTranslate2/tests/lm_fusion_test.cc  # 필요 시 신규
```

작업:

- fake scorer unit test 추가
- KenLM fixture integration test 추가
- baseline/fusion latency 비교
- HF POC 결과와 방향성 비교

## 18. 빌드 예시

KenLM 경로는 실제 환경에 맞게 조정한다.

```bash
cmake -S . -B build-lm \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_KENLM=ON \
  -DKENLM_ROOT=/tmp/kenlm/build \
  -DBUILD_TESTS=ON

cmake --build build-lm -j
./build-lm/tests/ctranslate2_test
```

Python local install:

```bash
pip install -e ./python
```

## 19. Rollback / feature flag

기본값에서는 기능이 꺼져 있다.

```text
lm_fusion_model_path empty
or lm_fusion_alpha <= 0
```

1차 구현에서는 LM 로드 실패 시 silent fallback하지 않는다. 실험 지표 오염을 막기 위해 명확히 실패시킨다.

운영 hardening 단계에서만 service-layer fallback을 추가한다.

## 20. Open Questions

1. KenLM을 CT2 build에 어떤 방식으로 포함할지 확정해야 한다.
   - system install
   - `KENLM_ROOT`
   - third_party vendoring

2. scorer를 매 `generate()`마다 만들지, replica별 cache로 둘지 결정해야 한다.
   - 실험 1차는 단순 생성 가능
   - 운영은 path별 cache가 맞다

3. previous-text prompt를 LM state에 포함할지 결정해야 한다.
   - 1차는 decode prefix 기준
   - faster-whisper의 previous text prompt를 쓰는 운영 경로라면 명시적으로 replay 필요

4. timestamp-enabled decoding에서 fusion을 허용할지 결정해야 한다.
   - 1차는 허용하되 timestamp token은 skip
   - timestamp rules가 먼저 적용되어야 함

## 21. 완료 조건

CT2 이식 실험 준비 완료 기준:

- `WITH_KENLM=ON`으로 CTranslate2가 빌드된다.
- Python `Whisper.generate()`가 LM fusion 옵션을 받는다.
- 기본 옵션에서는 기존 CT2 출력이 바뀌지 않는다.
- fake scorer unit test가 candidate selection과 beam reorder를 검증한다.
- 작은 샘플에서 CT2 fusion이 HF POC와 같은 방향의 용어 recall 개선을 보인다.
- p95 latency와 KenLM query count가 문서화된다.
