# CT2 KenLM BPE Fusion Design

이 문서는 CTranslate2 Whisper KenLM BPE shallow fusion 작업의 설계 SSOT이다.
구현 중 정책, 계약, 지원 범위가 바뀌면 이 문서를 먼저 갱신한다.

Reference 문서는 근거와 과거 초안이다. 최신 판단이 reference와 충돌하면 이 문서가 우선한다.

- `dev-docs/reference/implementation_plan_ct2.md`
- `dev-docs/reference/CT2 Whisper KenLM BPE Fusion 구현 계획.md`
- `dev-docs/reference/CTranslate2 KenLM BPE shallow fusion 코드 통합 리서치.md`

## 1. Goal

HuggingFace POC에서 검증한 Whisper BPE token-id KenLM 1-pass shallow fusion을 CTranslate2 Whisper beam search 내부로 이식한다.

Fusion score는 다음으로 정의한다.

```text
fused_score = CT2_ASR_logprob + beam_cumulative_score + alpha * KenLM_BPE_logprob_ln
```

KenLM은 Whisper token id를 pseudo-word로 변환한 corpus로 학습한다.

```text
Whisper token id 1234 -> KenLM word "t1234"
```

1차 구현의 목적은 도메인 long-tail 용어 recall 개선 가능성을 CT2 C++ 경로에서 검증하는 것이다.

## 2. Non-Goals

- Full-vocabulary shallow fusion은 1차 범위가 아니다.
- union/domain-successor 후보 확장은 1차 범위가 아니다.
- random sampling fusion은 1차 범위가 아니다.
- `return_alternatives`와 LM fusion 조합은 1차 범위가 아니다.
- Python per-step callback 기반 fusion은 사용하지 않는다.
- CT2 model conversion이나 model binary format은 바꾸지 않는다.
- KenLM 학습은 CT2 안에서 하지 않는다. Runtime은 `.binary` artifact만 로드한다.

## 3. Design Decisions

### 3.1 Fusion Mode

1차 구현은 `topk_strict`만 지원한다.

각 active beam row에서 ASR top `lm_fusion_asr_topk` 후보만 가져오고, 그 후보 안에서 KenLM 점수를 더해 batch top `num_candidates`를 다시 고른다.

ASR top-k 밖 token은 어떤 LM 점수로도 rescue하지 않는다. 첫 subtoken rescue가 필요하면 이후 union/domain-successor phase에서 다룬다.

### 3.2 Insertion Point

LM fusion은 `src/decoding.cc::BeamSearch::search()` 내부에 둔다.

삽입 지점은 다음 순서에서 `LogSoftMax`와 beam cumulative score broadcast가 끝난 직후, 기존 `reshape + sampler + unflatten_ids` 직전이다.

```text
decoder(...)
LogitsProcessor
disable_tokens.apply()
LogSoftMax or BiasedDecoder
beam cumulative score add
LM fusion candidate selection
EOS/finished 처리
active_beams 계산
decoder.update_state(...)
```

`LogitsProcessor`에는 넣지 않는다. `LogitsProcessor`는 beam별 external LM state를 소유하고 candidate selection 이후 state reorder를 따라가기 어렵다.

### 3.3 API Scope

Public API는 Whisper 쪽에만 노출한다.

Internal decoding layer는 generic scorer interface를 받되, translator/generator/greedy 경로로 의미가 번지지 않게 기본값은 완전 disabled로 둔다.

지원 조합:

- Whisper beam search
- `beam_size > 1`
- deterministic sampler path
- `return_alternatives == false`

Unsupported 조합은 조용히 baseline으로 떨어지지 않고 validation error를 낸다. 실험 지표 오염을 막기 위해 1차 구현에서는 명시 실패가 기본이다.

CT2에서 deterministic sampler path는 현재 `sampling_topk == 1` 또는 `sampling_temperature == 0`일 때 `BestSampler`가 선택되는 경로다. LM fusion은 random sampling path와 함께 쓰지 않는다.

### 3.4 Output ID Policy

CT2 decoder는 `update_output_layer()` 호출 후 output id와 original token id가 달라질 수 있다.

KenLM은 original Whisper token id 기준으로 학습되므로 LM scoring 전에 반드시 변환한다.

```text
output id -> original Whisper token id -> KenLM word id
```

Padding output id는 scoring 대상이 아니다. 현재 CT2는 padding slot을 original token `0`으로 역매핑할 수 있으므로, `Decoder`에 유효 output size와 padding output id를 구분하는 accessor를 추가해야 한다.

정책:

- `output_id < effective_output_size`: valid output id
- `output_id >= effective_output_size`: padding output id, candidate에서 제외
- `topk_ids`에는 기존 CT2 계약대로 output id 또는 flattened output id를 유지
- KenLM scoring에만 original id를 사용

### 3.5 Token Policy

Whisper text token만 LM state를 advance하고 LM score를 더한다.

Whisper에서는 `_eot_id` 미만을 text token으로 본다.

`original_id >= _eot_id`인 token은 skip token이다.

Skip token 정책:

```text
LM additive score = 0
candidate next state = previous LM state
```

이 정책은 `<|endoftext|>`, language token, task token, no-speech token, no-timestamps token, timestamp token에 적용된다.

### 3.6 LM State Policy

LM state는 `BeamSearch::search()` request local 상태다.

State order는 CT2 beam order와 동일하게 유지한다.

```text
lm_states shape: cur_batch_size * beam_size
index: batch_index * beam_size + beam_index
```

Candidate state는 candidate selection 결과와 같은 order를 가진다.

```text
candidate_lm_states shape: cur_batch_size * num_candidates
index: batch_index * num_candidates + candidate_index
```

`active_beams`로 beam gather를 할 때 LM state도 같은 index로 gather한다.

`non_finished_index`로 finished batch를 제거할 때 LM state도 같은 batch order로 prune한다.

### 3.7 Hard Prefix Policy

Hard prefix는 candidate selection 이후 `update_sample_with_prefix()`가 token과 gather index를 덮어쓸 수 있다.

따라서 hard-prefix forced step에서는 fusion score와 candidate next state를 미리 확정하면 state drift가 날 수 있다.

1차 정책:

- `use_hard_prefix`이고 live batch 중 하나라도 `step <= prefix_ids[batch_offset[i]].size()`이면 해당 decoding step 전체에서 LM fusion scoring을 우회한다.
- 기존 CT2 sampler/prefix update 경로로 최종 token을 확정한다.
- `update_sample_with_prefix()` 이후 최종 `topk_ids`와 `gather_indices` 기준으로 candidate LM state만 advance 또는 copy한다.
- batch별 prefix length가 달라도 1차 구현은 step 전체 우회를 사용한다. per-batch fusion/baseline 혼합은 후속 최적화로 둔다.

### 3.8 Prompt Replay Policy

Whisper prompt 중 decoder state에 이미 replay된 prefix text가 있으면 LM state도 같은 text history로 seed되어야 한다.

1차 구현에서는 제한된 prompt replay만 지원한다.

- `WhisperReplica::generate()`가 `start_tokens`와 분리한 prompt prefix를 기준으로 LM initial history를 전달한다.
- LM initial history에는 `original_id < _eot_id`인 text token만 넣는다.
- Whisper control token, task token, language token, timestamp token은 initial history에서 제외한다.

이 범위가 불명확한 previous-text prompt 운영 경로는 별도 parity test로 고정하기 전까지 open risk로 둔다.

### 3.9 KenLM Ownership

KenLM model은 immutable shared object로 보고, per-request/per-beam `State`는 공유하지 않는다.

1차 실험 구현은 단순 load도 가능하지만, 운영형 설계는 path-keyed shared cache가 맞다.

Cache key는 최소한 canonical path를 포함해야 하며, 운영 hardening에서는 file size/mtime 또는 artifact version을 추가한다.

### 3.10 Build Policy

KenLM은 optional dependency다.

기본 빌드는 기존 CT2와 동일해야 한다.

```text
WITH_KENLM=OFF
```

`WITH_KENLM=OFF` 빌드에서 LM fusion이 요청되면 명확한 error를 낸다.

```text
KenLM fusion requires CTranslate2 built with WITH_KENLM=ON
```

### 3.11 Score Semantics

Fusion enabled 상태에서 `return_scores=true`이면 반환 score는 ASR-only score가 아니라 fused cumulative score다.

`return_logits_vocab`는 기존 의미를 유지한다. 반환 logits/log-probs는 LM fusion 전 CT2 model 출력이다.

No-speech probability에는 LM fusion을 적용하지 않는다.

## 4. Public API

`models::WhisperOptions`에 다음 필드를 추가한다.

```cpp
std::string lm_fusion_model_path;
float lm_fusion_alpha = 0;
size_t lm_fusion_asr_topk = 50;
bool lm_fusion_debug = false;
```

Python `Whisper.generate()` keyword도 같은 이름으로 노출한다.

```python
lm_fusion_model_path: Optional[str] = None
lm_fusion_alpha: float = 0
lm_fusion_asr_topk: int = 50
lm_fusion_debug: bool = False
```

Feature off 조건:

```text
lm_fusion_model_path empty or lm_fusion_alpha <= 0
```

## 5. Internal API

Internal API는 KenLM 구현을 직접 BeamSearch에 노출하지 않는다.

권장 타입:

```cpp
struct LmFusionOptions {
  float alpha = 0;
  size_t asr_topk = 50;
  size_t text_token_limit = 0;
  bool debug = false;
};

class LmStateBatch {
public:
  virtual ~LmStateBatch() = default;
  virtual size_t size() const = 0;
  virtual void resize(size_t size) = 0;
  virtual std::unique_ptr<LmStateBatch> clone_empty(size_t size) const = 0;
};

class LmFusionScorer {
public:
  virtual ~LmFusionScorer() = default;
  virtual std::unique_ptr<LmStateBatch> make_initial_states(size_t size) const = 0;
  virtual float score_token(const LmStateBatch& in_states,
                            size_t in_index,
                            size_t original_token_id,
                            LmStateBatch& out_states,
                            size_t out_index) const = 0;
  virtual void copy_state(const LmStateBatch& in_states,
                          size_t in_index,
                          LmStateBatch& out_states,
                          size_t out_index) const = 0;
  virtual void gather(const LmStateBatch& src,
                      const std::vector<int32_t>& indices,
                      LmStateBatch& dst) const = 0;
  virtual void keep_batches(const LmStateBatch& src,
                            const std::vector<int32_t>& kept_batch_ids,
                            dim_t beam_size,
                            LmStateBatch& dst) const = 0;
};
```

State는 opaque type이므로 gather/prune은 scorer interface가 담당한다. 핵심 계약은 CT2 `active_beams`와 `non_finished_index`를 그대로 따르는 것이다.

## 6. Validation

LM fusion enabled일 때 validation 조건:

```text
alpha > 0
asr_topk > 0
scorer != nullptr
beam_size > 1
deterministic sampler path
return_alternatives == false
asr_topk <= vocabulary_size
```

Unsupported mode는 silent fallback하지 않는다.

## 7. Test Policy

우선순위가 가장 높은 test:

- padding output id가 LM scoring 대상에서 제외된다.
- fusion off에서 기존 CT2 결과가 변하지 않는다.
- ASR top-k 내부의 LM-preferred token이 winner로 올라온다.
- ASR top-k 밖 token은 rescue되지 않는다.
- special/timestamp token은 LM score 0, state copy다.
- beam reorder 후 다음 step LM state alignment가 유지된다.
- finished batch prune 후 LM state order가 유지된다.
- hard prefix forced step에서 최종 token과 LM state가 일치한다.
- valid candidate가 `num_candidates`보다 부족하면 silent fallback하지 않고 명확히 실패한다.
- `WITH_KENLM=OFF`에서 fusion 요청은 명확히 실패한다.

## 8. Open Questions

- 1차 구현에서 path-keyed KenLM cache를 포함할지, 단순 load로 시작할지 결정이 필요하다.
- previous-text prompt replay 범위를 실제 faster-whisper 운영 경로와 맞출지 확인이 필요하다. 1차 구현은 forwarded prompt prefix의 text token replay까지만 지원한다.
- timestamp-enabled decoding을 1차부터 허용할지, skip token 정책만으로 충분한지 test가 필요하다.
- KenLM을 system install, `KENLM_ROOT`, vendoring 중 어떤 방식으로 빌드에 포함할지 결정이 필요하다.
