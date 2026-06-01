# Whisper Phrase Bias Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CTranslate2의 Whisper 디코더에 도메인 용어 phrase bias를 추가해, 도메인 키워드를 모델 로드(도메인) 시 받아 디코딩 logits에 bias를 거는 기능을 구현한다.

**Architecture:** 기존 `LogitsProcessor` 파이프라인에 `PhraseBiasProcessor`를 additive하게 주입한다. 공개 모델 API(`ctranslate2::models::WhisperOptions`)에 phrase bias 옵션을 추가하고, 저수준 프로세서(`decoding_utils`)는 plain 타입만 받아 레이어링을 지킨다. block 모드는 기존 `DisableTokens`를 재사용해 CPU/GPU 모두 동작한다.

**Tech Stack:** C++17, Google Test(`tests/ctranslate2_test`), pybind11(Python binding), CMake.

> **이 플랜의 범위 = Phase 1 (block 모드 MVP, device-portable).** soft bias / reverse trie / CUDA indexed_add / Python path compiler / faster-whisper 연동은 §"Deferred Phases"에 별도 플랜으로 분리. 근거: `dev-docs/SSOT.md`(특히 §4 데이터 모델, §7 디자인 검증 V1~V5).

---

## 사전 확인 (코딩 시작 전 1회)

- [ ] **개인 fork 브랜치 준비**

```bash
cd /data/MyProject/stt/CTranslate2
git remote -v          # origin = YuBeomGon/CTranslate2 확인
git checkout -b feature/whisper-phrase-bias
```

- [ ] **C++ 테스트 빌드가 되는지 baseline 확인**

```bash
mkdir -p build && cd build
cmake -DBUILD_TESTS=ON -DWITH_CUDA=OFF -DWITH_MKL=OFF ..
make -j"$(nproc)" ctranslate2_test
./tests/ctranslate2_test ../tests/data
```
Expected: 기존 테스트 전부 PASS (phrase bias 추가 전 baseline).

> 빌드 옵션(`-DWITH_CUDA`, `-DWITH_MKL` 등)은 로컬 환경에 맞게 조정. Phase 1은 CPU만으로 검증 가능.

---

## File Structure

| 파일 | 책임 | 변경 |
|------|------|------|
| `include/ctranslate2/models/whisper.h` | Whisper 공개 옵션 API | `PhraseBiasMode`/`PhraseBiasPath`/`PhraseBias` 타입 + `WhisperOptions.phrase_biases` 추가 |
| `include/ctranslate2/decoding_utils.h` | 저수준 LogitsProcessor 선언 | `PhraseBiasEntry`(plain) + `PhraseBiasProcessor` 선언 |
| `src/decoding_utils.cc` | LogitsProcessor 구현 | `PhraseBiasProcessor::apply`(block) 구현 |
| `src/models/whisper.cc` | WhisperOptions→DecodingOptions 변환 + 프로세서 주입 | `models::PhraseBias`→`PhraseBiasEntry` 변환, `logits_processors`에 주입 |
| `python/cpp/whisper.cc` | Python binding | `phrase_biases` kwarg 파싱 → `WhisperOptions` |
| `tests/decoding_test.cc` | C++ 단위 테스트 | `PhraseBiasProcessor` 테스트 추가 (CMake 수정 불필요) |
| `python/tests/test_whisper_phrase_bias.py` | Python 통합 테스트 | 신규 |

**레이어링 원칙(V5/CONTRIBUTING 준수):** `decoding_utils.h`는 `models/whisper.h`를 include하지 않는다. 프로세서는 plain `PhraseBiasEntry`만 알고, `models::PhraseBias`→`PhraseBiasEntry` 변환은 `whisper.cc`가 담당한다. (기존 `WhisperOptions.suppress_tokens`(int) → `DecodingOptions.disable_ids` 변환과 동일한 패턴.)

---

## Task 1: Whisper 공개 옵션에 phrase bias 데이터 모델 추가

**Files:**
- Modify: `include/ctranslate2/models/whisper.h:11-59` (`WhisperOptions` 구조체 위에 타입 추가, 구조체 안에 필드 추가)
- Test: `tests/decoding_test.cc` (구조체가 컴파일/기본값을 만족하는지)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/decoding_test.cc` 맨 아래에 추가:

```cpp
#include <ctranslate2/models/whisper.h>

TEST(PhraseBiasTest, WhisperOptionsDefaultsEmpty) {
  ctranslate2::models::WhisperOptions options;
  EXPECT_TRUE(options.phrase_biases.empty());
}

TEST(PhraseBiasTest, PhraseBiasPathDefaults) {
  ctranslate2::models::PhraseBiasPath path;
  EXPECT_EQ(path.ids.size(), 0u);
  EXPECT_FLOAT_EQ(path.start_bias, 0.f);
  EXPECT_FLOAT_EQ(path.step_bias, 0.f);
  EXPECT_EQ(path.min_prefix_len, 1u);
  EXPECT_EQ(path.mode, ctranslate2::models::PhraseBiasMode::Soft);
}
```

- [ ] **Step 2: 테스트 빌드해서 실패 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test`
Expected: 컴파일 실패 — `'PhraseBiasPath' is not a member of 'ctranslate2::models'`.

- [ ] **Step 3: 최소 구현 — 타입과 필드 추가**

`include/ctranslate2/models/whisper.h`에서 `struct WhisperOptions {` 바로 위에 추가:

```cpp
    enum class PhraseBiasMode : int8_t {
      Soft = 0,   // logits += bias (Phase 2)
      Block = 1,  // disable next token (Phase 1)
    };

    // One tokenization path of a surface form. The bias is applied to the next
    // token when the current sequence suffix matches a prefix of `ids`.
    struct PhraseBiasPath {
      std::vector<size_t> ids;
      float start_bias = 0.f;        // bias for the first token of the path
      float step_bias = 0.f;         // bias for continuation tokens
      uint16_t min_prefix_len = 1;
      PhraseBiasMode mode = PhraseBiasMode::Soft;
    };

    // A surface form (and its aliases) compiled to one or more token paths.
    struct PhraseBias {
      std::vector<PhraseBiasPath> token_paths;
    };
```

그리고 `WhisperOptions` 구조체 맨 아래(`suppress_tokens` 필드 다음)에 추가:

```cpp
      // Phrase-level bias on domain terms. Empty = disabled (no overhead).
      std::vector<PhraseBias> phrase_biases;
```

`<cstdint>` 가 필요하면 파일 상단 include에 추가 (`int8_t`/`uint16_t`용).

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test --gtest_filter='PhraseBiasTest.*'`
Expected: `WhisperOptionsDefaultsEmpty`, `PhraseBiasPathDefaults` PASS.

- [ ] **Step 5: 커밋**

```bash
git add include/ctranslate2/models/whisper.h tests/decoding_test.cc
git commit -m "feat(whisper): add phrase bias data model to WhisperOptions"
```

---

## Task 2: PhraseBiasProcessor (block 모드) 구현

저수준 프로세서. plain `PhraseBiasEntry`를 받아, 현재 시퀀스 suffix가 path prefix와 일치하면 `mode=block`인 path의 다음 토큰을 disable 한다. 동작은 `SuppressSequences::apply`(`src/decoding_utils.cc`)의 suffix 매칭(`std::equal(end-N, end, ...)`)을 일반화한 것. soft entry가 들어오면 Phase 1에서는 명시적으로 throw 한다(Phase 2에서 제거).

**Files:**
- Modify: `include/ctranslate2/decoding_utils.h` (`SuppressSequences` 선언 근처, ~136행)
- Modify: `src/decoding_utils.cc` (`SuppressSequences::apply` 구현 근처)
- Test: `tests/decoding_test.cc`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/decoding_test.cc`에 추가. `sequences`는 디코드 루프에서 merged `[batch, length]` int32 레이아웃(V3)이며, step 0엔 null일 수 있음(V4):

```cpp
TEST(PhraseBiasTest, BlockDisablesNextTokenOnSuffixMatch) {
  // vocab_size = 5, batch = 1. logits 모두 1.0.
  StorageView logits({1, 5}, std::vector<float>(5, 1.0f));
  DisableTokens disable_tokens(logits, /*disable_value=*/0.0f);

  // path ids = [3, 4] block: 직전 토큰이 3이면 다음 토큰 4를 막는다.
  PhraseBiasEntry entry;
  entry.ids = {3, 4};
  entry.block = true;
  PhraseBiasProcessor processor({entry});

  // 현재 시퀀스 = [3] (마지막 토큰이 3)
  StorageView sequences({1, 1}, std::vector<int32_t>{3});
  processor.apply(/*step=*/1, logits, disable_tokens, sequences, {0}, nullptr);
  disable_tokens.apply();

  StorageView expected({1, 5}, std::vector<float>{1, 1, 1, 1, 0});  // token 4 막힘
  expect_storage_eq(logits, expected);
}

TEST(PhraseBiasTest, BlockNoMatchLeavesLogitsUnchanged) {
  StorageView logits({1, 5}, std::vector<float>(5, 1.0f));
  DisableTokens disable_tokens(logits, 0.0f);
  PhraseBiasEntry entry; entry.ids = {3, 4}; entry.block = true;
  PhraseBiasProcessor processor({entry});

  StorageView sequences({1, 1}, std::vector<int32_t>{2});  // 마지막 토큰 2 != 3
  processor.apply(1, logits, disable_tokens, sequences, {0}, nullptr);
  disable_tokens.apply();

  StorageView expected({1, 5}, std::vector<float>(5, 1.0f));
  expect_storage_eq(logits, expected);
}

TEST(PhraseBiasTest, BlockNullSequencesIsNoOp) {
  StorageView logits({1, 5}, std::vector<float>(5, 1.0f));
  DisableTokens disable_tokens(logits, 0.0f);
  PhraseBiasEntry entry; entry.ids = {3, 4}; entry.block = true;
  PhraseBiasProcessor processor({entry});

  StorageView empty;  // step 0: sequences 없음 (V4)
  processor.apply(0, logits, disable_tokens, empty, {0}, nullptr);
  disable_tokens.apply();

  StorageView expected({1, 5}, std::vector<float>(5, 1.0f));
  expect_storage_eq(logits, expected);
}

TEST(PhraseBiasTest, SoftEntryThrowsInPhase1) {
  PhraseBiasEntry entry; entry.ids = {3, 4}; entry.block = false;
  EXPECT_THROW(PhraseBiasProcessor({entry}), std::invalid_argument);
}
```

- [ ] **Step 2: 테스트 빌드해서 실패 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test`
Expected: 컴파일 실패 — `'PhraseBiasEntry'/'PhraseBiasProcessor' was not declared`.

- [ ] **Step 3: 헤더 선언 추가**

`include/ctranslate2/decoding_utils.h`의 `SuppressSequences` 클래스 선언 바로 위에 추가:

```cpp
  // Low-level, model-agnostic representation of one compiled phrase-bias path.
  struct PhraseBiasEntry {
    std::vector<size_t> ids;
    float start_bias = 0.f;
    float step_bias = 0.f;
    uint16_t min_prefix_len = 1;
    bool block = false;          // true = Block mode, false = Soft mode (Phase 2)
  };

  // Apply phrase-level bias on domain terms. Phase 1: block mode only.
  class PhraseBiasProcessor : public LogitsProcessor {
  public:
    explicit PhraseBiasProcessor(std::vector<PhraseBiasEntry> entries);

    bool apply_first() const override {
      return false;   // V1: run after GetNoSpeechProbs so no-speech prob is not polluted
    }

    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    std::vector<PhraseBiasEntry> _block_entries;
  };
```

`<cstdint>` 가 아직 없으면 추가.

- [ ] **Step 4: 구현 추가**

`src/decoding_utils.cc`의 `SuppressSequences::apply` 구현 바로 아래에 추가. 매칭 로직은 `SuppressSequences`와 동일하게 각 batch row의 suffix를 본다:

```cpp
  PhraseBiasProcessor::PhraseBiasProcessor(std::vector<PhraseBiasEntry> entries) {
    for (auto& entry : entries) {
      if (!entry.block)
        throw std::invalid_argument("PhraseBiasProcessor: soft mode is not implemented yet "
                                    "(Phase 2). Only block-mode entries are supported.");
      if (entry.ids.size() < 2)
        throw std::invalid_argument("PhraseBiasProcessor: block path must have at least 2 ids "
                                    "(prefix + token to disable).");
      _block_entries.emplace_back(std::move(entry));
    }
  }

  void PhraseBiasProcessor::apply(dim_t,
                                  StorageView&,
                                  DisableTokens& disable_tokens,
                                  const StorageView& sequences,
                                  const std::vector<dim_t>&,
                                  const std::vector<std::vector<size_t>>*) {
    if (!sequences)        // V4: step 0 has no generated tokens yet
      return;

    const dim_t batch_size = sequences.dim(0);
    const dim_t length = sequences.dim(1);

    for (dim_t batch_id = 0; batch_id < batch_size; ++batch_id) {
      const auto* begin = sequences.index<int32_t>({batch_id, 0});
      const auto* end = begin + length;

      for (const auto& entry : _block_entries) {
        const dim_t compare_length = static_cast<dim_t>(entry.ids.size()) - 1;
        if (length < compare_length)
          continue;

        const bool match = std::equal(end - compare_length,
                                      end,
                                      entry.ids.begin(),
                                      entry.ids.begin() + compare_length);
        if (match)
          disable_tokens.add(batch_id, entry.ids.back());
      }
    }
  }
```

`<stdexcept>` include 가 없으면 `src/decoding_utils.cc` 상단에 추가.

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test --gtest_filter='PhraseBiasTest.*'`
Expected: 4개 테스트 모두 PASS.

- [ ] **Step 6: 커밋**

```bash
git add include/ctranslate2/decoding_utils.h src/decoding_utils.cc tests/decoding_test.cc
git commit -m "feat(decoding): add PhraseBiasProcessor with block mode"
```

---

## Task 3: WhisperReplica::generate 에 프로세서 주입

`models::PhraseBias` → `PhraseBiasEntry` 변환 후 `decoding_options.logits_processors`에 추가. no-speech / timestamp 프로세서 등록 다음에 둔다(V1 — apply_first=false 그룹). 옵션이 비면 프로세서를 만들지 않는다(D7).

**Files:**
- Modify: `src/models/whisper.cc:296-345` (`DecodingOptions` 구성 블록, timestamp 프로세서 emplace 직후)
- Test: `tests/decoding_test.cc` (변환 헬퍼 단위 테스트) + Task 5 Python 통합 테스트

- [ ] **Step 1: 실패하는 테스트 작성 — 변환 헬퍼**

변환 로직을 테스트 가능한 free 함수로 분리한다. `tests/decoding_test.cc`에 추가:

```cpp
TEST(PhraseBiasTest, ConvertModelOptionToEntriesBlockOnly) {
  using namespace ctranslate2::models;
  std::vector<PhraseBias> biases(1);
  PhraseBiasPath p;
  p.ids = {7, 8, 9};
  p.mode = PhraseBiasMode::Block;
  biases[0].token_paths.push_back(p);

  std::vector<ctranslate2::PhraseBiasEntry> entries = to_phrase_bias_entries(biases);
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].ids, (std::vector<size_t>{7, 8, 9}));
  EXPECT_TRUE(entries[0].block);
}
```

- [ ] **Step 2: 빌드해서 실패 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test`
Expected: 컴파일 실패 — `'to_phrase_bias_entries' was not declared`.

- [ ] **Step 3: 변환 헬퍼 선언/구현**

`include/ctranslate2/decoding_utils.h`의 `PhraseBiasProcessor` 선언 다음에 free 함수 선언 추가 (저수준 헤더가 models 타입을 모르므로, 변환 함수는 models 쪽에 두는 게 레이어링상 맞다 → 대신 `include/ctranslate2/models/whisper.h`에 선언):

`include/ctranslate2/models/whisper.h`의 `PhraseBias` 정의 다음에 추가:

```cpp
    // Flatten model-level phrase biases into low-level processor entries.
    std::vector<PhraseBiasEntry>
    to_phrase_bias_entries(const std::vector<PhraseBias>& phrase_biases);
```

`whisper.h` 상단에 `#include "ctranslate2/decoding_utils.h"` 추가(이미 있으면 생략). `PhraseBiasEntry`는 `ctranslate2` 네임스페이스이므로 `models` 안에서는 `ctranslate2::PhraseBiasEntry`로 참조.

`src/models/whisper.cc` 상단(익명 네임스페이스 또는 `models` 네임스페이스 내부)에 구현 추가:

```cpp
    std::vector<PhraseBiasEntry>
    to_phrase_bias_entries(const std::vector<PhraseBias>& phrase_biases) {
      std::vector<PhraseBiasEntry> entries;
      for (const auto& bias : phrase_biases) {
        for (const auto& path : bias.token_paths) {
          PhraseBiasEntry entry;
          entry.ids = path.ids;
          entry.start_bias = path.start_bias;
          entry.step_bias = path.step_bias;
          entry.min_prefix_len = path.min_prefix_len;
          entry.block = (path.mode == PhraseBiasMode::Block);
          entries.emplace_back(std::move(entry));
        }
      }
      return entries;
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test --gtest_filter='PhraseBiasTest.ConvertModelOptionToEntriesBlockOnly'`
Expected: PASS.

- [ ] **Step 5: generate 경로에 프로세서 주입**

`src/models/whisper.cc`의 timestamp 프로세서 `emplace_back(... ApplyTimestampRules ...)` 블록 **직후**(약 345행, `decode(...)` 호출 전)에 추가:

```cpp
      if (!options.phrase_biases.empty()) {
        auto entries = to_phrase_bias_entries(options.phrase_biases);
        if (!entries.empty())
          decoding_options.logits_processors.emplace_back(
            std::make_shared<PhraseBiasProcessor>(std::move(entries)));
      }
```

- [ ] **Step 6: 전체 C++ 테스트 통과 확인 (회귀 없음)**

Run: `cd build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test ../tests/data`
Expected: 기존 테스트 + PhraseBiasTest 전부 PASS.

- [ ] **Step 7: 커밋**

```bash
git add include/ctranslate2/models/whisper.h src/models/whisper.cc tests/decoding_test.cc
git commit -m "feat(whisper): inject PhraseBiasProcessor into generate logits pipeline"
```

---

## Task 4: Python binding 에 phrase_biases kwarg 추가

Python에서 `phrase_biases`를 dict 리스트로 받아 `models::WhisperOptions.phrase_biases`로 파싱. Phase 1 스키마: `[{"token_paths": [{"ids": [int...], "mode": "block", "min_prefix_len": int}]}]`. `start_bias`/`step_bias`도 파싱하되 block에선 미사용.

**Files:**
- Modify: `python/cpp/whisper.cc:32-78` (`generate` 시그니처 + 옵션 구성), `:242-258` (`py::arg` 등록)
- Test: Task 5의 pytest

- [ ] **Step 1: generate 시그니처에 인자 추가**

`python/cpp/whisper.cc`의 `WhisperWrapper::generate(...)` 파라미터 목록 `suppress_tokens` 다음에 추가:

```cpp
               const std::optional<py::list>& phrase_biases,
```

상단 include에 `<pybind11/stl.h>` 가 이미 있는지 확인(없으면 추가).

- [ ] **Step 2: 파싱 로직 추가**

`options.suppress_tokens` 설정 블록(약 68-71행) 다음에 추가:

```cpp
        if (phrase_biases) {
          for (const auto& bias_obj : *phrase_biases) {
            auto bias_dict = bias_obj.cast<py::dict>();
            models::PhraseBias bias;
            for (const auto& path_obj : bias_dict["token_paths"].cast<py::list>()) {
              auto path_dict = path_obj.cast<py::dict>();
              models::PhraseBiasPath path;
              for (const auto& id : path_dict["ids"].cast<py::list>())
                path.ids.push_back(id.cast<size_t>());
              if (path_dict.contains("start_bias"))
                path.start_bias = path_dict["start_bias"].cast<float>();
              if (path_dict.contains("step_bias"))
                path.step_bias = path_dict["step_bias"].cast<float>();
              if (path_dict.contains("min_prefix_len"))
                path.min_prefix_len = path_dict["min_prefix_len"].cast<uint16_t>();
              const std::string mode =
                path_dict.contains("mode") ? path_dict["mode"].cast<std::string>() : "soft";
              path.mode = (mode == "block") ? models::PhraseBiasMode::Block
                                            : models::PhraseBiasMode::Soft;
              bias.token_paths.push_back(std::move(path));
            }
            options.phrase_biases.push_back(std::move(bias));
          }
        }
```

- [ ] **Step 3: py::arg 등록**

`.def("generate", ...)`의 `py::arg("suppress_blank")=true,` 다음에 추가:

```cpp
             py::arg("phrase_biases")=py::none(),
```

- [ ] **Step 4: Python wheel 빌드**

```bash
cd /data/MyProject/stt/CTranslate2/build
make -j"$(nproc)"
sudo make install && sudo ldconfig
cd ../python
CTRANSLATE2_ROOT=/usr/local pip install -e .
```
Expected: 빌드 성공, `import ctranslate2` 에러 없음.

- [ ] **Step 5: 커밋**

```bash
git add python/cpp/whisper.cc
git commit -m "feat(python): expose phrase_biases kwarg in Whisper.generate"
```

---

## Task 5: Python 통합 테스트 (block 모드 end-to-end)

block phrase bias가 실제로 특정 토큰 생성을 막는지 검증. 변환된 작은 Whisper 모델이 필요. `tests/data`에 whisper 모델이 있으면 그것을, 없으면 `ct2-transformers-converter`로 `openai/whisper-tiny`를 변환해 사용.

**Files:**
- Create: `python/tests/test_whisper_phrase_bias.py`

- [ ] **Step 1: 테스트 작성**

```python
import os
import numpy as np
import pytest
import ctranslate2

WHISPER_MODEL = os.environ.get("CT2_WHISPER_TINY")  # ct2로 변환된 whisper-tiny 경로

pytestmark = pytest.mark.skipif(
    not WHISPER_MODEL or not os.path.isdir(WHISPER_MODEL),
    reason="set CT2_WHISPER_TINY to a converted whisper-tiny model dir",
)


def _features():
    # 80-mel, 3000 frames (30s) zero input — 디코딩이 돌기만 하면 됨
    return ctranslate2.StorageView.from_array(np.zeros((1, 80, 3000), dtype=np.float32))


def test_block_phrase_bias_suppresses_token():
    model = ctranslate2.models.Whisper(WHISPER_MODEL, device="cpu")
    prompt = [model.hf_tokenizer_prefix_tokens()] if False else [[50258, 50259, 50359, 50363]]

    base = model.generate(_features(), prompt, beam_size=1)
    base_tokens = base[0].sequences_ids[0]
    assert len(base_tokens) >= 2

    # baseline 출력의 (n-1번째 -> n번째) 전이를 block 한다.
    a, b = base_tokens[0], base_tokens[1]
    biased = model.generate(
        _features(), prompt, beam_size=1,
        phrase_biases=[{"token_paths": [{"ids": [a, b], "mode": "block"}]}],
    )
    biased_tokens = biased[0].sequences_ids[0]

    # 같은 prefix(a) 뒤에 b가 그대로 다시 나오면 안 된다.
    assert not (biased_tokens[0] == a and biased_tokens[1] == b)


def test_empty_phrase_biases_is_noop():
    model = ctranslate2.models.Whisper(WHISPER_MODEL, device="cpu")
    prompt = [[50258, 50259, 50359, 50363]]
    a = model.generate(_features(), prompt, beam_size=1)[0].sequences_ids[0]
    b = model.generate(_features(), prompt, beam_size=1, phrase_biases=[])[0].sequences_ids[0]
    assert a == b
```

> 프롬프트 토큰(`[50258, ...]`)은 모델 vocab에 맞게 조정. `_features()`는 디코더가 돌게 하는 더미. 실제 음성 검증은 `dev-docs/testing-manual.md`(별도)에서 수행.

- [ ] **Step 2: 모델 준비 후 테스트 실행**

```bash
pip install transformers[torch]
ct2-transformers-converter --model openai/whisper-tiny \
  --output_dir /tmp/whisper-tiny-ct2 --copy_files tokenizer.json preprocessor_config.json
CT2_WHISPER_TINY=/tmp/whisper-tiny-ct2 pytest python/tests/test_whisper_phrase_bias.py -v
```
Expected: `test_block_phrase_bias_suppresses_token`, `test_empty_phrase_biases_is_noop` PASS.

- [ ] **Step 3: 커밋**

```bash
git add python/tests/test_whisper_phrase_bias.py
git commit -m "test(python): block-mode phrase bias integration test"
```

---

## Task 6: 문서 갱신 + 푸시

- [ ] **Step 1: changes.md 에 구현 내역 기록**

`dev-docs/changes.md` 최상단에 추가:

```markdown
### 2026-06-XX — Phase 1: block 모드 phrase bias 구현
- 무엇을: WhisperOptions.phrase_biases + PhraseBiasProcessor(block) + Python binding + 테스트
- 어디: whisper.h, decoding_utils.h/.cc, whisper.cc, python/cpp/whisper.cc, tests/
- 영향: 옵션 비면 no-op(D7). soft/trie/CUDA는 Phase 2+
```

- [ ] **Step 2: 개인 repo 푸시**

```bash
git push -u origin feature/whisper-phrase-bias
```

---

## Self-Review (작성자 체크 완료)

- **Spec 커버리지**: SSOT §2 D1(additive 주입)=Task3, D2(block)=Task2, D6(apply_first=false)=Task2 Step3, D7(빈 옵션 no-op)=Task3 Step5+Task5. §4 데이터모델=Task1. §7 V1/V3/V4=Task2~3 반영. soft(D2 일부)/trie(D3)/indexed_add/path compiler(D4)/도메인 baking(D8)=Deferred로 명시 분리.
- **Placeholder 스캔**: 모든 코드 step에 실제 코드 포함. "적절히 처리" 류 없음.
- **타입 일관성**: `PhraseBiasEntry`(plain, `ctranslate2::`), `PhraseBias`/`PhraseBiasPath`/`PhraseBiasMode`(`ctranslate2::models::`), `to_phrase_bias_entries` 변환 — Task1~4 전체에서 동일 시그니처 사용 확인.

---

## Deferred Phases (각각 별도 플랜으로 작성 예정)

> 각 Phase는 독립적으로 동작/테스트 가능한 단위. Phase 1 완료 후 SSOT 기준으로 별도 impl-plan 작성.

- **Phase 2 — Soft bias + `indexed_add` primitive**: `RepetitionPenalty::apply` 템플릿(V5) 따라 `ops::Gather`+`DEVICE_AND_TYPE_DISPATCH`로 logits in-place 수정. `primitives.h`에 `indexed_add` 선언 + `src/cpu/primitives.cc`/`src/cuda/primitives.cu` 구현. PhraseBiasProcessor에서 soft entry throw 제거하고 start/step_bias 스케줄 적용. 같은 `(batch,token)` 다중 bias dedupe/reduce.
- **Phase 3 — Reverse trie + tail cache**: naïve suffix scan(Phase 1~2)을 reverse trie로 교체. per-step 비용을 path 수가 아닌 tail 길이에 비례시킴. `CompiledPhraseBias`(immutable, `shared_ptr<const>`)로 모델 로드(도메인) 시 1회 빌드(D8).
- **Phase 4 — Python tokenizer path compiler + YAML**: surface→token_paths 컴파일(canonical + alias variants + bounded DP). `phrase_bias_config="domain_terms.yaml"` 진입점. path explosion 캡(max_terms/max_paths_per_term/max_path_len).
- **Phase 5 — faster-whisper / WhisperLiveKit 연동**: 도메인별 모델 init 시 vocab baking. 별도 repo 작업.
- **검증** — `dev-docs/testing-manual.md`(서브에이전트 작성)로 vanilla vs fork A/B, 도메인 음성 로그 회귀.
