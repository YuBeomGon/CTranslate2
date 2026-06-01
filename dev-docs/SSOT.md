# SSOT — Whisper Phrase Bias 포크 (Single Source of Truth)

> 이 문서가 이 포크의 **단일 진실 공급원**입니다. 충돌/혼선 시 이 문서를 기준으로 삼습니다.
> 상세 근거와 분석은 [`deep-research-report.md`](deep-research-report.md) 참고. 이 SSOT는 그 보고서를 실행 가능한 사양으로 distill한 것입니다.
>
> **상태**: 설계 확정, 구현 전 (last updated: 2026-06-01)
> **개발 워크플로우**: 개인 fork(`YuBeomGon/CTranslate2`)에서 `feature/whisper-phrase-bias` 브랜치로 작업 → 개인 repo push. upstream 동기화는 [`upstream-sync.md`](upstream-sync.md).

---

## 1. 목적

CTranslate2의 **Whisper 디코더에 도메인 용어 phrase bias**를 추가한다.

- **positive bias** → 도메인 정답 용어(예: "트랜스포머")의 recall ↑
- **negative bias** → 자주 발생하는 오인식(예: "트랜스퍼머") 억제 ↓

faster-whisper의 `hotwords`는 prompt/hint 성격이라 next-token score를 직접 못 건드린다.
정확한 score-level 제어를 위해 **CT2 C++ 코어에 phrase-level bias를 추가**한다 (monkey patch 아님).

## 2. 핵심 설계 결정 (확정)

| # | 결정 | 이유 |
|---|------|------|
| D1 | 기존 `LogitsProcessor` 파이프라인에 `PhraseBiasProcessor` **additive 주입** | Whisper generate가 이미 no-speech/timestamp를 이 방식으로 주입. 모델 포맷 변경 불필요 |
| D2 | **두 모드**: `block`(기존 `DisableTokens`/`SuppressSequences` 재사용) + `soft`(새 `indexed_add`, `+= delta`) | block은 MVP로 빠르게, soft는 정밀 제어 |
| D3 | 매칭은 **reverse trie + tail cache**. naïve scan은 POC(≤128 path)만, automaton은 5000+ path일 때만 | per-step 비용을 path 수가 아닌 tail 길이에 비례 |
| D4 | **토크나이저 분리**: surface→`token_paths` 컴파일은 Python에서. CT2엔 token ID path만 전달 | byte-level BPE variability를 C++ 밖에서 흡수. 성능·유지보수 ↑ |
| D5 | C++ 코어 수정 후 **private wheel/Docker 배포** (`X.Y.Z+pb1` 식 suffix). rollback = 플래그 off 또는 upstream wheel | 운영 현실성, 손쉬운 롤백 |
| D6 | `PhraseBiasProcessor::apply_first()` → **`false`** | step 0의 `GetNoSpeechProbs`가 먼저 실행되어야 no-speech 확률 오염 방지 |
| D7 | 옵션 비었으면 processor **생성 안 함** | disabled 시 오버헤드 ≈ 0, 후방 호환 보장 |
| D8 | **컴파일은 모델 로드(=도메인/고객사)당 1회** (chunk/generate마다 절대 X). 모델 init 때 그 도메인 vocab을 token_paths 토큰화 + reverse trie 빌드하고, 결과(`CompiledPhraseBias`)를 모든 generate에서 **read-only 참조로 재사용** | 도메인/고객사별로 모델이 따로 로딩됨 → vocab을 모델에 baking하는 게 자연스러움. trie는 immutable → `shared_ptr<const>`로 thread-safe 공유 |
| D9 | **동적 키워드 변경은 MVP 범위 밖.** 도메인 vocab 변경 시 해당 도메인 모델 재로드. 추후 `set_phrase_biases()` 검토 | streaming 중간 변경·thread-safety 복잡도 회피 |

## 3. 수정 대상 파일 맵

| 파일 | 변경 내용 |
|------|-----------|
| `include/ctranslate2/models/whisper.h` | `PhraseBiasMode`, `PhraseBiasPath`, `PhraseBias`, `WhisperOptions.phrase_biases` 추가 |
| `src/models/whisper.cc` | `WhisperReplica::generate`에서 `PhraseBiasProcessor`를 `decoding_options.logits_processors`에 연결 (no-speech 이후) |
| `include/ctranslate2/decoding_utils.h` | `PhraseBiasProcessor`, `SparseLogitsAdjustments` 선언 |
| `src/decoding_utils.cc` | reverse trie 빌드, per-step 매칭, block/soft apply 구현 |
| `include/ctranslate2/primitives.h` | `indexed_add` primitive 선언 |
| `src/cpu/primitives.cc` | CPU sparse add 구현 |
| `src/cuda/primitives.cu` | CUDA sparse add 구현 (dedupe 후 unique index 1회 add) |
| `python/cpp/whisper.cc` | `phrase_biases` kwargs ↔ `WhisperOptions` 매핑 |
| `python/setup.py` | 내부 wheel version suffix 정책 |
| `python/tests/`, `tests/` | unit + integration + A/B 테스트 |

## 4. 데이터 모델 (확정 API)

```cpp
// include/ctranslate2/models/whisper.h
enum class PhraseBiasMode : int8_t { Soft = 0, Block = 1 };

struct PhraseBiasPath {
  std::vector<size_t> ids;        // one tokenization path
  float start_bias = 0.f;         // first-token bias
  float step_bias = 0.f;          // continuation bias
  uint16_t min_prefix_len = 1;
  PhraseBiasMode mode = PhraseBiasMode::Soft;
};

struct PhraseBias {
  std::vector<PhraseBiasPath> token_paths;  // multi-path for one surface/alias set
};

struct WhisperOptions {
  // existing fields...
  std::vector<PhraseBias> phrase_biases;    // additive; empty = no-op
};

// 모델 로드(도메인)당 1회 빌드되는 immutable 컴파일 산출물 (D8). generate마다 재사용.
struct CompiledPhraseBias {
  ReverseTrie trie;
  size_t max_path_len = 0;
  // ... bias 메타. shared_ptr<const CompiledPhraseBias>로 공유 → read-only, thread-safe
};
```

**Bias 식**:
```text
matched_len = 현재 beam suffix ∩ path prefix 일치 길이
matched_len == 0                  → effective_bias = start_bias
matched_len >= min_prefix_len     → effective_bias = start_bias + (matched_len - min_prefix_len + 1) * step_bias
else                              → 0
```
- positive: `start_bias > 0` (rare proper noun nudging)
- negative: `start_bias = 0`, `step_bias < 0` (첫 토큰 과억제 방지)

**Python 표면 — 권장: init 시점 compile (D8)**:
```python
# 단일 vocab 배포: 모델 init에 baking. trie는 생성자에서 1회 빌드.
model = WhisperModel("large-v3-ct2", device="cuda",
                     phrase_bias_config="domain_terms.yaml")  # 또는 phrase_biases=[...]

# 이후 모든 generate는 컴파일된 trie만 참조 (재컴파일 X)
model.generate(features, prompts, beam_size=5)
```

CT2 low-level은 여전히 **numeric token_paths만** 받는다 (문자열 토큰화는 Python에서, D4). YAML/surface→`token_paths` 변환은 init 때 wrapper가 수행.

## 4.5 컴파일 수명주기 & 배포 모델 (D8/D9)

**배포 구조**: 모델은 **도메인/고객사 단위로 로딩**된다 (고객사 A → 모델 A, 고객사 B → 모델 B). 각 도메인 모델은 자기 vocab을 가진다.

**철칙**: token 토큰화 + trie 빌드는 **모델 로드(도메인)당 1회**. chunk/generate마다 재컴파일 금지.

| 배포 형태 | 권장 방식 |
|-----------|-----------|
| **도메인/고객사별 모델 로딩 (기본)** | 해당 도메인 vocab을 **모델 init에 baking**. init 때 1회 compile → `CompiledPhraseBias`를 replica가 보유 → 그 도메인의 모든 generate에서 재사용 |
| 한 프로세스가 여러 도메인 모델 보유 (멀티테넌트) | 문제 없음. 각 도메인 모델이 각자 baked vocab을 가지므로 baking 방식 그대로 |

> 도메인별 모델 로딩이라 "하나의 모델 인스턴스를 여러 vocab이 공유"하는 상황은 발생하지 않는다. 따라서 **baking이 메인 경로**다. (만약 미래에 단일 모델을 여러 도메인이 공유해야 하면, generate 옵션이 `shared_ptr<const CompiledPhraseBias>` 핸들도 받도록 확장 — 현재는 불필요.)

## 5. 가드레일 / 검증 규칙

- `ids` 빈 경로 금지 · `min_prefix_len < len(ids)` 권장
- `mode=block`에 positive bias 값 설정 금지
- `abs(start_bias)`, `abs(step_bias)` 상한 설정 (과도한 insertion/deletion 방지)
- special token(SOT/EOT/timestamp) 포함 금지
- 중복 path dedupe; 같은 `(batch,row,token)` 다중 bias는 apply 전 reduce
- **path explosion 캡** (init compile 시 enforce): `max_terms` 200~500, `max_paths_per_term` 8~16, `max_path_len` 상한
- **domain recall과 insertion ratio를 항상 함께 측정** (recall만 오르고 insertion 폭증 = 품질 악화)

## 6. 구현 순서 (로드맵)

1. **CT2 fork 준비** → `feature/whisper-phrase-bias` 브랜치, 개인 repo push
2. **옵션/스키마 확정** → `whisper.h` 데이터 모델(`PhraseBias*`, `CompiledPhraseBias`) + Python binding 인자
3. **block 모드 MVP** → suffix-prefix 로직·인터페이스 먼저 안정화 (기존 `DisableTokens` 재사용)
4. **init-time compile** → token_paths → reverse trie를 모델 로드(도메인) 시 1회 빌드, generate는 참조만 (D8)
5. **soft bias** → `indexed_add` primitive (CPU/CUDA) + sparse add
6. **tokenizer path compiler** (Python) + faster-whisper init config(YAML) 연동
7. **WhisperLiveKit/streaming init config 연결** + latency benchmark
8. **검증** → unit/integration, 도메인 음성 로그 A/B, 성능 회귀 CI

## 7. 디자인 검증 (2026-06-01, 실제 체크아웃 코드 대조)

보고서가 추정한 인터페이스를 현재 repo 코드로 1:1 확인함. **7개 가정 전부 일치.**

| 가정 | 실제 코드 | 결과 |
|------|-----------|------|
| `LogitsProcessor::apply(step, logits, disable_tokens, sequences, batch_offset, prefix)` + `apply_first()` 기본 false | `include/ctranslate2/decoding_utils.h:74-95` | ✅ 시그니처 정확히 일치 |
| `DisableTokens.add(batch_id, token_id)` / CPU 직접대입·GPU flat index 수집 | `decoding_utils.h:36-69` | ✅ block 모드는 이거 재사용 |
| Whisper generate가 `decoding_options.logits_processors.emplace_back(...)`로 주입 | `src/models/whisper.cc:325-337` | ✅ no-speech(`make_shared` 조건부)·timestamp 동일 패턴 |
| decode 루프가 processor 순회 후 `disable_tokens.apply()` | `src/decoding.cc:518-527` (beam), `:863-866` (greedy) | ✅ **루프 2개 모두** 동일 호출 → block/soft 둘 다 자동 적용 |
| `primitives::indexed_fill` 존재, `indexed_add` 없음 | `include/ctranslate2/primitives.h:21` | ✅ `indexed_add` 신규 추가 필요 (file map대로) |
| `SuppressSequences::apply`가 suffix 매칭(`std::equal(end-N, end, ...)`) | `src/decoding_utils.cc` | ✅ reverse trie는 이 패턴의 일반화 |
| `WhisperOptions` 구조체, `phrase_biases` 추가가 purely additive | `whisper.h:11-59` | ✅ 기존 필드 영향 없음 |

### 검증 중 발견 — 구현 시 반영할 refinement

| # | 발견 | 반영 |
|---|------|------|
| V1 | processor 순서가 `make_logits_processors`(`decoding.cc:1091`)에서 고정: **apply_first=true 그룹 → RepetitionPenalty → NoRepeatNgram → SuppressTokens → SuppressSequences → apply_first=false 그룹**. PhraseBias(`apply_first=false`)는 **맨 마지막**에 실행 | 좋음 — bias가 모든 suppression 위에 얹힘. `GetNoSpeechProbs`(apply_first=**true**, `whisper.cc:209`)는 항상 먼저 → **D6 가정 정확히 성립** |
| V2 | `disable_tokens.apply()`가 모든 processor 뒤 **1회** 실행 → 같은 토큰에 block(=lowest) 적용 시 soft 추가분을 덮어씀 | block이 soft를 이김 = 안전한 의미론. 의도된 동작 |
| V3 | apply() 안에서 `sequences`(alive_seq)는 **merged `[batch*beam, length]`** 레이아웃 (`decoding.cc:519` merge/split 래핑) | reverse trie는 각 row의 tail을 `SuppressSequences`처럼 읽으면 됨 |
| V4 | step 0엔 `sequences`가 null (`if (!sequences) return;` — RepetitionPenalty·SuppressSequences 공통 가드) | **PhraseBiasProcessor도 null 가드 필수.** null이면 matched_len=0 → path 첫 토큰 `start_bias`만 적용 |
| V5 | **soft bias 구현 템플릿 = `RepetitionPenalty::apply`** — `ops::Gather` + `DEVICE_AND_TYPE_DISPATCH`로 `primitives::penalize_previous_tokens` 호출해 logits in-place 수정 | `indexed_add`는 `penalize_previous_tokens`/`indexed_fill`의 sibling primitive로 추가. soft = logits 직접 수정, block = `disable_tokens.add()` |

> 결론: **재디자인 불필요.** SSOT 그대로 구현 가능. 위 V1~V5만 구현 플랜에 반영하면 됨.

## 8. 관련 문서

- [`deep-research-report.md`](deep-research-report.md) — 전체 설계 근거·분석·벤치마크 계획 (배경)
- [`changes.md`](changes.md) — upstream 대비 실제 변경 내역 로그
- [`upstream-sync.md`](upstream-sync.md) — OpenNMT 본가 동기화 절차
