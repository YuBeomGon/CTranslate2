# Phase 1 Implementation Plan — CPU Positive Phrase Bias + Reverse Trie

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (사용자 선택). 체크박스(`- [ ]`)로 추적, 각 Task 끝에 커밋.
> 결정·금지사항·exemplar는 `dev-docs/SSOT.md`가 기준. 이 플랜은 **P1만**의 실행 지시서.

**Goal:** CT2 Whisper 디코더에 CPU positive phrase bias를 추가한다 — 현재 시퀀스 suffix가 phrase prefix와 일치하면 다음 토큰 logit에 `step_bias`를 더한다. reverse trie로 매칭, `indexed_add` primitive(CPU)로 적용.

**Architecture:** 저수준 `PhraseBiasEntry`(token ids + 미리 계산된 step_bias)를 reverse trie에 빌드 → `PhraseBiasProcessor`(LogitsProcessor)가 step마다 각 row tail을 trie lookup → `(token,delta)` 합산·clamp → unique index로 `indexed_add(logits)`. soft bias는 항상 `indexed_add` primitive를 dispatch 경유로 호출(P2에서 CUDA만 추가하면 됨, 인터페이스 불변).

**Tech Stack:** C++17, Google Test, CMake. CPU/Ruy 빌드.

### P1 범위 (고정)
- **포함:** C++ core only · low-level token ids 입력 · step_bias **이미 계산됨 가정** · reverse trie · continuation-only(start_bias 없음) · overlap 합산+최종 clamp · `indexed_add` CPU primitive · empty no-op · `tests/` 단위 테스트(forced synthetic logits).
- **제외:** Python tokenizer compile · leading-space 실제 tokenizer 검증 · GPU indexed_add(P2) · faster-whisper(P4) · YAML(P3).

### 금지사항 (SSOT §7)
전역 boost 금지 · start token boost 금지 · 매 generate마다 trie 재생성 금지 · `phrase_biases` 비면 processor 생성 금지 · 새 CUDA 스타일 금지 · **`indexed_add`에 중복 index 금지(processor가 호출 전 dedupe/합산)**.

### 빌드/테스트 (SSOT §13)
- 설정됨: `build/` (`cmake -DBUILD_TESTS=ON -DBUILD_CLI=OFF -DWITH_MKL=OFF -DWITH_RUY=ON -DWITH_CUDA=OFF -DOPENMP_RUNTIME=NONE -DCMAKE_BUILD_TYPE=Release ..`)
- 빌드: `cd /data/MyProject/stt/CTranslate2/build && make -j"$(nproc)" ctranslate2_test`
- 실행(positional data-dir 인자 필요): `./tests/ctranslate2_test --gtest_filter='<F>' /tmp`
- **무시할 baseline 실패:** `CPU/OpDeviceFPTest.Gemm/GemmBias/GemmResidual /float32` (Ruy 아티팩트). 항상 필터로 본인 테스트만 확인.

---

## File Structure

| 파일 | 책임 |
|------|------|
| `include/ctranslate2/primitives.h` | `indexed_add` 선언 (`indexed_fill` 옆) |
| `src/cpu/primitives.cc` | `indexed_add` CPU 구현 + 인스턴스화 매크로 |
| `include/ctranslate2/decoding_utils.h` | `PhraseBiasEntry`, `PhraseBiasTrie`, `PhraseBiasProcessor` 선언 (plain 타입, models 의존 X) |
| `src/decoding_utils.cc` | trie build/lookup + processor apply |
| `src/models/whisper.cc` | `models::PhraseBias`→`PhraseBiasEntry` 변환 + `logits_processors` 주입(empty면 skip) |
| `tests/primitives_test.cc` | `indexed_add` 단위 테스트 |
| `tests/decoding_test.cc` | trie + processor 단위 테스트 (기존 PhraseBiasTest 옆) |

---

## Task 1: `indexed_add` CPU primitive

`indexed_fill`(set)을 복사해 `indexed_add`(+=)로. delta는 per-index 배열. **unique index 가정**(중복은 호출자가 미리 합산).

**Files:**
- Modify: `include/ctranslate2/primitives.h` (`indexed_fill` 선언 근처, ~21행)
- Modify: `src/cpu/primitives.cc` (`indexed_fill` 구현 ~60행, 인스턴스화 매크로 ~1168행)
- Test: `tests/primitives_test.cc`

- [ ] **Step 1: 실패 테스트 작성** — `tests/primitives_test.cc` 맨 아래에 추가:

```cpp
TEST(PrimitiveTest, IndexedAddCPU) {
  std::vector<float> x = {0, 1, 2, 3, 4};
  std::vector<float> deltas = {0.5f, 0.25f};
  std::vector<int32_t> indices = {1, 3};
  primitives<Device::CPU>::indexed_add(x.data(), deltas.data(), indices.data(), 2);
  std::vector<float> expected = {0, 1.5f, 2, 3.25f, 4};
  EXPECT_EQ(x, expected);
}
```

- [ ] **Step 2: 빌드해서 실패 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test`
Expected: 컴파일 실패 — `'indexed_add' is not a member of 'ctranslate2::primitives<...>'`.

- [ ] **Step 3: 선언 추가** — `include/ctranslate2/primitives.h`의 `indexed_fill` 선언 바로 아래:

```cpp
    template <typename T>
    static void indexed_add(T* x, const T* deltas, const int32_t* indices, dim_t num_indices);
```
(주변 선언이 `template <typename T> static void ...` 형식이면 그대로 맞춤. `indexed_fill` 선언이 클래스 템플릿 멤버면 그 형식을 따른다.)

- [ ] **Step 4: CPU 구현 추가** — `src/cpu/primitives.cc`의 `indexed_fill` 구현 바로 아래:

```cpp
  template <typename T>
  void primitives<Device::CPU>::indexed_add(T* x, const T* deltas,
                                            const int32_t* indices, dim_t num_indices) {
    for (dim_t i = 0; i < num_indices; ++i)
      x[indices[i]] += deltas[i];
  }
```

- [ ] **Step 5: 명시적 인스턴스화 추가** — `src/cpu/primitives.cc`에서 `indexed_fill` 인스턴스화 매크로 줄(~1168행) **바로 아래**에 같은 형식으로 (백슬래시 줄맞춤 유지):

```cpp
  template void                                                         \
  primitives<Device::CPU>::indexed_add(T*, const T*, const int32_t*, dim_t);
```
(해당 매크로가 `DECLARE_IMPL(T)` 식으로 T별 인스턴스화하는 블록이면 그 안에 `indexed_fill`과 나란히 추가한다. 링크 에러 나면 이 줄 빠진 것.)

- [ ] **Step 6: 통과 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test --gtest_filter='PrimitiveTest.IndexedAddCPU' /tmp`
Expected: PASS.

- [ ] **Step 7: 커밋**

```bash
git add include/ctranslate2/primitives.h src/cpu/primitives.cc tests/primitives_test.cc
git commit -m "feat(primitives): add indexed_add (CPU)"
```

---

## Task 2: `PhraseBiasEntry` + reverse trie

token-id path를 reverse-prefix trie로. lookup은 현재 row suffix를 보고 boost할 `(token, step)`를 **합산**해 반환. `min_prefix_len`은 build 시 필터(매칭 길이 < min_prefix_len인 continuation rule 제외).

**Files:**
- Modify: `include/ctranslate2/decoding_utils.h` (`SuppressSequences` 선언 근처)
- Modify: `src/decoding_utils.cc`
- Test: `tests/decoding_test.cc`

- [ ] **Step 1: 실패 테스트 작성** — `tests/decoding_test.cc`에 추가 (공유 prefix가 둘 다 반환되는지가 핵심):

```cpp
TEST(PhraseBiasTest, TrieSharedPrefixReturnsBoth) {
  PhraseBiasTrie trie;
  trie.add(PhraseBiasEntry{{1, 2, 3}, 0.25f, 1});  // [A=1,B=2,C=3]
  trie.add(PhraseBiasEntry{{1, 2, 4}, 0.25f, 1});  // [A=1,B=2,D=4]
  trie.add(PhraseBiasEntry{{8, 9}, 0.5f, 1});      // [X=8,Y=9]

  auto boost = [&](std::vector<int32_t> tail) {
    std::map<size_t, float> out;
    trie.lookup(tail.data(), static_cast<dim_t>(tail.size()), out);
    return out;
  };

  EXPECT_EQ(boost({1}), (std::map<size_t,float>{{2, 0.5f}}));        // [A] → B (두 path가 합산 0.25+0.25)
  EXPECT_EQ(boost({1, 2}), (std::map<size_t,float>{{3, 0.25f}, {4, 0.25f}}));  // [A,B] → C,D
  EXPECT_EQ(boost({8}), (std::map<size_t,float>{{9, 0.5f}}));        // [X] → Y
  EXPECT_TRUE(boost({7}).empty());                                  // 불일치 → 없음
  EXPECT_TRUE(boost({}).empty());                                   // suffix 없음 → 없음
}

TEST(PhraseBiasTest, TrieMinPrefixLenSkipsShortMatch) {
  PhraseBiasTrie trie;
  trie.add(PhraseBiasEntry{{1, 2, 3}, 0.25f, 2});  // min_prefix_len=2 → [A] 매칭(길이1)으로는 boost 안 함
  std::map<size_t, float> out1;
  trie.lookup(std::vector<int32_t>{1}.data(), 1, out1);
  EXPECT_TRUE(out1.empty());                        // [A]만으론 boost 없음
  std::map<size_t, float> out2;
  trie.lookup(std::vector<int32_t>{1, 2}.data(), 2, out2);
  EXPECT_EQ(out2, (std::map<size_t,float>{{3, 0.25f}}));  // [A,B] → C
}

TEST(PhraseBiasTest, TrieSkipsTooShortEntry) {
  PhraseBiasTrie trie;
  trie.add(PhraseBiasEntry{{5}, 0.5f, 1});  // 1-token → continuation 불가 → skip
  std::map<size_t, float> out;
  trie.lookup(std::vector<int32_t>{5}.data(), 1, out);
  EXPECT_TRUE(out.empty());
}
```

- [ ] **Step 2: 빌드해서 실패 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test`
Expected: 컴파일 실패 — `'PhraseBiasEntry'/'PhraseBiasTrie' was not declared`.

- [ ] **Step 3: 헤더 선언** — `include/ctranslate2/decoding_utils.h`의 `SuppressSequences` 선언 위에 추가 (`<cstdint>`, `<unordered_map>`, `<map>`, `<vector>` 필요 시 include):

```cpp
  // Low-level, model-agnostic phrase bias path. step_bias is precomputed (= total_bias/(len-1), clamped).
  struct PhraseBiasEntry {
    std::vector<size_t> ids;
    float step_bias = 0.f;
    uint16_t min_prefix_len = 1;
  };

  // Reverse-prefix trie: maps a reversed suffix to the next token(s) to boost.
  class PhraseBiasTrie {
  public:
    void add(const PhraseBiasEntry& entry);
    // out[token] += step for every continuation rule whose prefix matches the suffix of `tail`.
    void lookup(const int32_t* tail, dim_t tail_len, std::map<size_t, float>& out) const;

  private:
    struct Node {
      std::unordered_map<size_t, Node> children;     // key = previous token
      std::vector<std::pair<size_t, float>> actions; // (next_token, step) valid at this depth
    };
    Node _root;
  };
```

- [ ] **Step 4: 구현** — `src/decoding_utils.cc`에 추가:

```cpp
  void PhraseBiasTrie::add(const PhraseBiasEntry& entry) {
    const auto& ids = entry.ids;
    if (ids.size() < 2)
      return;  // 1-token phrase: continuation 불가 → skip
    // rule j (1..len-1): suffix [ids[0..j-1]] 일치 시 ids[j] boost.
    for (size_t j = 1; j < ids.size(); ++j) {
      if (j < entry.min_prefix_len)
        continue;  // 매칭 길이 j 가 min_prefix_len 미만이면 제외
      Node* node = &_root;
      // reverse-prefix: ids[j-1], ids[j-2], ..., ids[0]
      for (size_t k = j; k-- > 0; )
        node = &node->children[ids[k]];
      node->actions.emplace_back(ids[j], entry.step_bias);
    }
  }

  void PhraseBiasTrie::lookup(const int32_t* tail, dim_t tail_len,
                              std::map<size_t, float>& out) const {
    const Node* node = &_root;
    for (dim_t k = tail_len; k-- > 0; ) {
      auto it = node->children.find(static_cast<size_t>(tail[k]));
      if (it == node->children.end())
        break;
      node = &it->second;
      for (const auto& action : node->actions)
        out[action.first] += action.second;  // 합산 (overlap)
    }
  }
```

- [ ] **Step 5: 통과 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test --gtest_filter='PhraseBiasTest.Trie*' /tmp`
Expected: 3 PASS.

- [ ] **Step 6: 커밋**

```bash
git add include/ctranslate2/decoding_utils.h src/decoding_utils.cc tests/decoding_test.cc
git commit -m "feat(decoding): add PhraseBiasEntry and reverse trie"
```

---

## Task 3: `PhraseBiasProcessor` (CPU apply, forced synthetic logits)

trie lookup → 각 row의 `(token,delta)` 합산 → 최종 per-token clamp(`max_token_delta`, 기본 2.0) → unique `(flat_index, delta)` → `indexed_add`. `apply_first()=false`. step 0 `sequences` null 가드.

**Files:**
- Modify: `include/ctranslate2/decoding_utils.h`
- Modify: `src/decoding_utils.cc`
- Test: `tests/decoding_test.cc`

- [ ] **Step 1: 실패 테스트 작성** (모델 의존 X — 손으로 만든 logits):

```cpp
TEST(PhraseBiasTest, ProcessorBoostsNextTokenOnSuffixMatch) {
  // vocab=5, batch=1. logits 모두 0.
  StorageView logits({1, 5}, std::vector<float>(5, 0.f));
  DisableTokens disable(logits, std::numeric_limits<float>::lowest());

  std::vector<PhraseBiasEntry> entries = {{{1, 2, 3}, 0.25f, 1}};  // [A=1,B=2,C=3]
  PhraseBiasProcessor proc(entries);

  StorageView seq({1, 1}, std::vector<int32_t>{1});  // suffix [A]
  proc.apply(1, logits, disable, seq, {0}, nullptr);

  StorageView expected({1, 5}, std::vector<float>{0, 0, 0.25f, 0, 0});  // token 2(B) += 0.25
  expect_storage_eq(logits, expected);
}

TEST(PhraseBiasTest, ProcessorOverlapSumsAndClamps) {
  StorageView logits({1, 5}, std::vector<float>(5, 0.f));
  DisableTokens disable(logits, std::numeric_limits<float>::lowest());
  // 두 phrase 모두 [A]→B boost: 0.6 + 0.6 = 1.2 → clamp 1.0
  std::vector<PhraseBiasEntry> entries = {{{1, 2}, 0.6f, 1}, {{1, 2}, 0.6f, 1}};
  PhraseBiasProcessor proc(entries, /*max_token_delta=*/1.0f);
  StorageView seq({1, 1}, std::vector<int32_t>{1});
  proc.apply(1, logits, disable, seq, {0}, nullptr);
  StorageView expected({1, 5}, std::vector<float>{0, 0, 1.0f, 0, 0});
  expect_storage_eq(logits, expected);
}

TEST(PhraseBiasTest, ProcessorNoMatchAndNullAreNoOp) {
  std::vector<PhraseBiasEntry> entries = {{{1, 2, 3}, 0.25f, 1}};
  PhraseBiasProcessor proc(entries);
  // 불일치
  StorageView logits1({1, 5}, std::vector<float>(5, 0.f));
  DisableTokens d1(logits1, std::numeric_limits<float>::lowest());
  StorageView seq({1, 1}, std::vector<int32_t>{7});
  proc.apply(1, logits1, d1, seq, {0}, nullptr);
  expect_storage_eq(logits1, StorageView({1, 5}, std::vector<float>(5, 0.f)));
  // null sequences (step 0)
  StorageView logits2({1, 5}, std::vector<float>(5, 0.f));
  DisableTokens d2(logits2, std::numeric_limits<float>::lowest());
  StorageView empty;
  proc.apply(0, logits2, d2, empty, {0}, nullptr);
  expect_storage_eq(logits2, StorageView({1, 5}, std::vector<float>(5, 0.f)));
}
```

- [ ] **Step 2: 빌드해서 실패 확인** → `'PhraseBiasProcessor' was not declared`.

- [ ] **Step 3: 헤더 선언** — `decoding_utils.h`의 `PhraseBiasTrie` 아래:

```cpp
  // Positive continuation phrase bias. apply_first()=false so it runs after no-speech.
  class PhraseBiasProcessor : public LogitsProcessor {
  public:
    explicit PhraseBiasProcessor(const std::vector<PhraseBiasEntry>& entries,
                                 float max_token_delta = 2.0f);

    bool apply_first() const override { return false; }

    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    PhraseBiasTrie _trie;
    float _max_token_delta;
  };
```

- [ ] **Step 4: 구현** — `src/decoding_utils.cc`. StorageView/dispatch 사용은 **`RepetitionPenalty::apply`(같은 파일)의 패턴을 그대로 따른다** (device/dtype 처리):

```cpp
  PhraseBiasProcessor::PhraseBiasProcessor(const std::vector<PhraseBiasEntry>& entries,
                                           float max_token_delta)
    : _max_token_delta(max_token_delta) {
    for (const auto& entry : entries)
      _trie.add(entry);  // 1-token/min_prefix_len 필터는 trie.add 내부
  }

  void PhraseBiasProcessor::apply(dim_t,
                                  StorageView& logits,
                                  DisableTokens&,
                                  const StorageView& sequences,
                                  const std::vector<dim_t>&,
                                  const std::vector<std::vector<size_t>>*) {
    if (!sequences)
      return;  // step 0: 생성된 토큰 없음

    const dim_t batch_size = logits.dim(0);
    const dim_t vocab_size = logits.dim(-1);
    const dim_t length = sequences.dim(1);

    std::vector<int32_t> flat_indices;
    std::vector<float> deltas;
    for (dim_t b = 0; b < batch_size; ++b) {
      const int32_t* row = sequences.index<int32_t>({b, 0});
      std::map<size_t, float> boost;
      _trie.lookup(row, length, boost);  // 합산
      for (const auto& kv : boost) {
        const float delta = std::min(kv.second, _max_token_delta);  // 최종 clamp
        flat_indices.push_back(static_cast<int32_t>(b * vocab_size + kv.first));
        deltas.push_back(delta);
      }
    }
    if (flat_indices.empty())
      return;

    const Device device = logits.device();
    const DataType dtype = logits.dtype();
    StorageView indices({static_cast<dim_t>(flat_indices.size())}, flat_indices);
    StorageView delta_view({static_cast<dim_t>(deltas.size())}, deltas);
    indices = indices.to(device);
    delta_view = delta_view.to(device).to(dtype);
    DEVICE_AND_TYPE_DISPATCH(
      device, dtype,
      primitives<D>::indexed_add(logits.data<T>(), delta_view.data<T>(),
                                 indices.data<int32_t>(), indices.size()));
  }
```
(`<map>`, `<algorithm>` include 확인. `src/dispatch.h`의 `DEVICE_AND_TYPE_DISPATCH`가 이미 이 파일에서 쓰이는지 보고 없으면 include.)

- [ ] **Step 5: 통과 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test --gtest_filter='PhraseBiasTest.Processor*' /tmp`
Expected: 3 PASS.

- [ ] **Step 6: 커밋**

```bash
git add include/ctranslate2/decoding_utils.h src/decoding_utils.cc tests/decoding_test.cc
git commit -m "feat(decoding): add PhraseBiasProcessor (CPU continuation bias)"
```

---

## Task 4: generate 주입 + empty no-op

`models::PhraseBias`(whisper.h, `PhraseBiasPath{ids, step_bias, ...}`) → `PhraseBiasEntry` 변환 후 `decoding_options.logits_processors`에 주입(no-speech 등록 뒤). `phrase_biases` 비면 주입 안 함.

**Files:**
- Modify: `include/ctranslate2/models/whisper.h` (변환 헬퍼 선언)
- Modify: `src/models/whisper.cc` (변환 구현 + 주입)
- Test: `tests/decoding_test.cc`

- [ ] **Step 1: 실패 테스트 작성** — 변환 헬퍼 단위 테스트:

```cpp
#include <ctranslate2/models/whisper.h>

TEST(PhraseBiasTest, ConvertModelOptionToEntries) {
  using namespace ctranslate2::models;
  std::vector<PhraseBias> biases(1);
  PhraseBiasPath p;
  p.ids = {1, 2, 3};
  p.step_bias = 0.25f;
  p.min_prefix_len = 1;
  biases[0].token_paths.push_back(p);

  std::vector<ctranslate2::PhraseBiasEntry> entries = to_phrase_bias_entries(biases);
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].ids, (std::vector<size_t>{1, 2, 3}));
  EXPECT_FLOAT_EQ(entries[0].step_bias, 0.25f);
  EXPECT_EQ(entries[0].min_prefix_len, 1u);
}
```

- [ ] **Step 2: 빌드해서 실패 확인** → `'to_phrase_bias_entries' was not declared`.

- [ ] **Step 3: 변환 헬퍼 선언** — `include/ctranslate2/models/whisper.h` (상단에 `#include "ctranslate2/decoding_utils.h"` 추가, `PhraseBias` 정의 다음):

```cpp
    std::vector<PhraseBiasEntry>
    to_phrase_bias_entries(const std::vector<PhraseBias>& phrase_biases);
```
(`PhraseBiasEntry`는 `ctranslate2` 네임스페이스 → `models` 안에서 `ctranslate2::PhraseBiasEntry`로 참조하거나 `using`.)

- [ ] **Step 4: 변환 구현** — `src/models/whisper.cc` (`models` 네임스페이스 내):

```cpp
    std::vector<PhraseBiasEntry>
    to_phrase_bias_entries(const std::vector<PhraseBias>& phrase_biases) {
      std::vector<PhraseBiasEntry> entries;
      for (const auto& bias : phrase_biases) {
        for (const auto& path : bias.token_paths) {
          PhraseBiasEntry entry;
          entry.ids = path.ids;
          entry.step_bias = path.step_bias;
          entry.min_prefix_len = path.min_prefix_len;
          entries.emplace_back(std::move(entry));
        }
      }
      return entries;
    }
```

- [ ] **Step 5: 변환 테스트 통과 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test --gtest_filter='PhraseBiasTest.ConvertModelOptionToEntries' /tmp`
Expected: PASS.

- [ ] **Step 6: generate 주입** — `src/models/whisper.cc`의 timestamp 프로세서 `emplace_back(... ApplyTimestampRules ...)` 블록 **직후**(약 345행, `decode(...)` 호출 전)에 추가:

```cpp
      if (!options.phrase_biases.empty()) {
        auto entries = to_phrase_bias_entries(options.phrase_biases);
        if (!entries.empty())
          decoding_options.logits_processors.emplace_back(
            std::make_shared<PhraseBiasProcessor>(entries));
      }
```

- [ ] **Step 7: 전체 회귀 확인 (empty no-op 포함)**

Run: `cd build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test ../tests/data`
Expected: 기존 테스트 + 모든 PhraseBiasTest + PrimitiveTest.IndexedAddCPU PASS. (baseline Gemm 3개 실패는 무시 — 우리와 무관.)

- [ ] **Step 8: 커밋**

```bash
git add include/ctranslate2/models/whisper.h src/models/whisper.cc tests/decoding_test.cc
git commit -m "feat(whisper): inject PhraseBiasProcessor into generate (empty = no-op)"
```

---

## Self-Review

**Spec 커버리지 (SSOT 대조):** continuation-only positive bias=Task3 · start_bias 없음(첫 토큰 boost 안 함)=trie가 j≥1만 rule 생성(Task2) · step_bias 정밀=Task3 logits +=step · reverse trie=Task2 · overlap 합산+clamp=Task3 ProcessorOverlapSumsAndClamps · indexed_add primitive(P1 CPU, dispatch 경유, unique index)=Task1+Task3 · empty no-op=Task4 Step6/7 · 1-token skip & min_prefix_len=Task2 · forced synthetic logits 테스트=Task3 · 금지(전역 boost·start boost·trie 재생성·중복 index)=trie 구조+processor 합산으로 충족. 범위 외(tokenizer/GPU/faster-whisper/YAML)=미포함.

**Placeholder 스캔:** 모든 코드 step에 실제 코드. "적절히 처리" 류 없음. StorageView/dispatch는 `RepetitionPenalty::apply` 패턴 참조 명시.

**타입 일관성:** `PhraseBiasEntry{ids, step_bias, min_prefix_len}`, `PhraseBiasTrie::{add,lookup}`, `PhraseBiasProcessor(entries, max_token_delta)`, `to_phrase_bias_entries`, `indexed_add(T*, const T*, const int32_t*, dim_t)` — Task 1~4 전체 동일 시그니처.

**주의(실행자):** `whisper.h`의 `PhraseBiasPath`에 `step_bias` 필드가 이미 있는지 확인(Task 1(이전) commit 34e52dd에서 추가됨). `to_phrase_bias_entries`는 `step_bias`를 그대로 복사(총합→step 분배는 P3). StorageView 생성자/`.to()` 형식은 실제 헤더와 `RepetitionPenalty::apply` 사용례에 맞춰 조정.

## 다음 (P1 이후)
- **P2** GPU `indexed_add` (CUDA, parity) — 별도 plan
- **P3** CT2 Python 바인딩만 (ids+bias, **토크나이저 없음** — SSOT §0.1) — 별도 plan
- **P4** faster-whisper 연동 — 별도 plan
