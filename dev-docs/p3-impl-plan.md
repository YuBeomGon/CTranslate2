# Phase 3 Implementation Plan — CT2 Python Binding + **Load-time Persistent Trie**

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (또는 subagent-driven-development). 체크박스(`- [ ]`)로 추적, 각 Task 끝에 커밋.
> 결정·범위·금지사항은 `dev-docs/SSOT.md`가 기준. 이 플랜은 **P3만**. P1(CPU)·P2(GPU) 완료됨.

**Goal:** faster-whisper가 compile한 `phrase_biases`(token ids + step_bias)를 CT2 `Whisper` **생성자**에 넘기면, CT2가 reverse trie를 **1회 build해 persistent 보관**하고, 매 generate에서 **rebuild 없이 재사용**하도록 만든다. + per-call `generate(phrase_biases=...)` override도 지원.

**Architecture:** trie를 **`WhisperWrapper`(per-Whisper-object, pybind 래퍼)에 `shared_ptr<const PhraseBiasTrie>`로 보관** → generate마다 `WhisperOptions.compiled_phrase_bias_trie`로 **read-only 주입**. replica는 shared_ptr를 읽기만 → ReplicaPool race 없음, lock 불필요. 생성자 주입이라 immutable(mutable setter 보류). C++ 코어는 `build_phrase_bias_trie(entries)→shared_ptr` + trie를 공유하는 processor ctor를 추가하고, `WhisperReplica::generate`가 compiled trie 있으면 그걸로(rebuild 없음) processor 생성.

**Tech Stack:** C++17, pybind11, Google Test, pytest. (효과 통합 테스트만 transformers/torch + whisper-tiny.)

### ⚠️ 아키텍처 경계 (SSOT §0.1) — P3는 토크나이저를 만들지 않는다
- **CT2엔 토크나이저 없음.** P3는 ids+bias를 받아 trie를 build/보관하는 **C++/통로**만. 문자열→ids(leading-space 2 path, special 제거, roundtrip)는 **P4 faster-whisper**. P3 테스트에 tokenizer correctness(SSOT cat 1) 없음.

### 왜 persistent trie인가 (SSOT §1, P1/P2 deviation 닫기)
- 현재 P1 구현은 `WhisperReplica::generate`에서 **매 call `make_shared<PhraseBiasProcessor>(entries)` → trie rebuild**. SSOT §1 "build once, generate마다 재컴파일 금지"를 어김.
- P3에서 trie를 load-time 1회 build → 매 generate는 포인터 참조만. 스트리밍(잦은 generate) overhead 제거. (~100 term은 forward가 압도하지만 설계상 닫는다.)

### P3 범위 (고정)
- **포함:** (C++) `build_phrase_bias_trie` + shared-trie processor ctor + `WhisperOptions.compiled_phrase_bias_trie` + generate에서 cached trie 사용 · (Python) `PhraseBiasPath/PhraseBias` 노출 + `Whisper(phrase_biases=...)` 생성자 주입 + `generate(phrase_biases=...)` per-call override + 통합 테스트.
- **제외:** tokenizer compile(P4) · faster-whisper(P4) · `start_bias`/`Block` Python 노출(parked) · **mutable `set_phrase_biases()`(보류)** · YAML/config 파일 · CUDA python ext(CPU로 충분).

### 금지사항 (SSOT §7 관련)
- Python에서 문자열 키워드 받지 말 것 — **ids만**.
- `phrase_biases` None/빈 → trie 보관 안 함, options 주입 안 함 → **processor 생성 안 됨**(empty no-op, §7-5).
- compiled trie는 **immutable**(`shared_ptr<const>`) — replica가 절대 수정 안 함. mutable setter 추가 금지(보류).
- parked 필드(`start_bias`/`mode`) Python 노출 금지.

### 빌드/테스트 환경
- C++ lib `build/`(CPU, P1/P2). C++ 테스트는 `build/tests/ctranslate2_test`.
- Python ext는 install된 lib에 링크(Task 3). 모든 python 실행에 `LD_LIBRARY_PATH`(install-cpu/lib[64]) 필요.
- 효과 통합 테스트는 `transformers`+`torch`+네트워크(whisper-tiny). 없으면 skip(바인딩 단위가 1차 게이트).
- known-fail: C++ baseline Gemm 3개(SSOT §13) 외 신규 0.

### 노출할 Python API (확정)
```python
import ctranslate2
P = ctranslate2.models.PhraseBiasPath
B = ctranslate2.models.PhraseBias

biases = [                                            # faster-whisper가 compile한 결과
    B(token_paths=[
        P(ids=[100, 200, 300], step_bias=0.25),       # " 트랜스포머"
        P(ids=[101, 200, 300], step_bias=0.25),       # "트랜스포머"
    ]),
]
# 권장(상용): 생성자 주입 (1회 compile → persistent trie)
model = ctranslate2.models.Whisper("model_dir", device="cpu", phrase_biases=biases)

# generate의 phrase_biases semantics (3-way, 확정):
model.generate(features, prompts)                              # = None: model-level cached trie 사용
model.generate(features, prompts, phrase_biases=None)         # 동일: model-level
model.generate(features, prompts, phrase_biases=[])          # [] : 이 호출만 bias 끔 (disable)
model.generate(features, prompts, phrase_biases=other_biases) # [...] : per-call override (실험/테스트용, 매 call build)
```

> **None vs [] (확정):** `None`=생성자 model-level trie 사용 · `[]`=이 generate만 disable(ablation 스위치) · `[...]`=per-call override. pybind `optional<vector>`가 None(nullopt)/[](빈 vector)를 구분하므로 구현 가능.
> **재사용/성능:** model-level trie 재사용은 **구조로 보장**(WhisperWrapper가 `_compiled_trie` 1회 build·보유, generate는 read-only 참조). 테스트는 "재빌드 안 함"을 직접 증명하지 않고 **output stability**만 확인. **per-call override(`[...]`)는 매 call trie build 허용 = 실험/테스트 전용**, 상용 경로는 생성자 `phrase_biases`만 쓴다.

---

## File Structure

| 파일 | 책임 | Task |
|------|------|------|
| `include/ctranslate2/decoding_utils.h` | `build_phrase_bias_trie` 선언 + `PhraseBiasProcessor(shared_ptr<const PhraseBiasTrie>)` ctor | T1 |
| `src/decoding_utils.cc` | 위 구현 (기존 `(entries)` ctor는 delegating) | T1 |
| `tests/decoding_test.cc` | shared-trie 공유 테스트 | T1 |
| `include/ctranslate2/models/whisper.h` | `WhisperOptions.compiled_phrase_bias_trie` | T2 |
| `src/models/whisper.cc` | generate: compiled trie 우선, 없으면 entries fallback | T2 |
| `python/cpp/whisper.cc` | `PhraseBiasPath/PhraseBias` 노출 + `WhisperWrapper._compiled_trie` + 생성자 kwarg + generate 주입 | T4,T5,T6 |
| `python/ctranslate2/models/__init__.py` | 타입 re-export | T5 |
| `python/tests/test_phrase_bias.py` (신규) | 단위 + 통합 | T5,T6,T7 |

---

## Task 1: C++ — shared trie (`build_phrase_bias_trie` + processor ctor)

trie를 `shared_ptr<const PhraseBiasTrie>`로 공유 가능하게. 기존 `(entries)` ctor는 새 ctor로 delegate → P1 테스트 안 깨짐.

**Files:** `include/ctranslate2/decoding_utils.h`, `src/decoding_utils.cc`, `tests/decoding_test.cc`

- [ ] **Step 1: 실패 테스트 작성** — `tests/decoding_test.cc`의 마지막 PhraseBias 테스트 뒤에:

```cpp
TEST(PhraseBiasTest, SharedTrieReusedByProcessors) {
  std::vector<PhraseBiasEntry> entries = {{{1, 2, 3}, 0.25f, 1}};
  std::shared_ptr<const PhraseBiasTrie> trie = build_phrase_bias_trie(entries);
  ASSERT_TRUE(trie != nullptr);

  // 같은 trie를 공유하는 두 processor가 동일하게 동작.
  for (int i = 0; i < 2; ++i) {
    StorageView logits({1, 5}, std::vector<float>(5, 0.f));
    DisableTokens disable(logits, std::numeric_limits<float>::lowest());
    PhraseBiasProcessor proc(trie);  // shared_ptr ctor
    StorageView seq({1, 1}, std::vector<int32_t>{1});
    proc.apply(1, logits, disable, seq, {0}, nullptr);
    StorageView expected({1, 5}, std::vector<float>{0, 0, 0.25f, 0, 0});
    expect_storage_eq(logits, expected);
  }
}
```

- [ ] **Step 2: 빌드해서 실패 확인**

Run: `cd /data/MyProject/stt/CTranslate2/build && make -j"$(nproc)" ctranslate2_test 2>&1 | grep -iE "build_phrase_bias_trie|error" | head`
Expected: 컴파일 실패 — `build_phrase_bias_trie` 미선언 / `PhraseBiasProcessor` shared_ptr ctor 없음.

- [ ] **Step 3: 헤더 선언** — `include/ctranslate2/decoding_utils.h`. `PhraseBiasProcessor` 선언을 아래로 교체(shared_ptr ctor 추가, 멤버 타입 변경), 그리고 클래스 위에 free 함수 선언:

```cpp
  // Build an immutable, shareable trie from entries (call once at load time).
  std::shared_ptr<const PhraseBiasTrie>
  build_phrase_bias_trie(const std::vector<PhraseBiasEntry>& entries);

  // Positive continuation phrase bias. apply_first()=false so it runs after no-speech.
  class PhraseBiasProcessor : public LogitsProcessor {
  public:
    // Convenience: build a fresh trie from entries.
    explicit PhraseBiasProcessor(const std::vector<PhraseBiasEntry>& entries,
                                 float max_token_delta = 2.0f);
    // Share a pre-built trie (no rebuild) — used by the load-time cached path.
    explicit PhraseBiasProcessor(std::shared_ptr<const PhraseBiasTrie> trie,
                                 float max_token_delta = 2.0f);

    bool apply_first() const override { return false; }

    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    std::shared_ptr<const PhraseBiasTrie> _trie;
    float _max_token_delta;
  };
```
(`<memory>`는 storage_view.h 경유로 들어오지만 명시 include 권장: 상단 `#include <memory>`.)

- [ ] **Step 4: 구현** — `src/decoding_utils.cc`. 기존 `PhraseBiasProcessor` 생성자/`apply`의 `_trie.` 사용을 아래로 교체:

```cpp
  std::shared_ptr<const PhraseBiasTrie>
  build_phrase_bias_trie(const std::vector<PhraseBiasEntry>& entries) {
    auto trie = std::make_shared<PhraseBiasTrie>();
    for (const auto& entry : entries)
      trie->add(entry);  // 1-token/min_prefix_len 필터는 trie.add 내부
    return trie;
  }

  PhraseBiasProcessor::PhraseBiasProcessor(const std::vector<PhraseBiasEntry>& entries,
                                           float max_token_delta)
    : PhraseBiasProcessor(build_phrase_bias_trie(entries), max_token_delta) {}

  PhraseBiasProcessor::PhraseBiasProcessor(std::shared_ptr<const PhraseBiasTrie> trie,
                                           float max_token_delta)
    : _trie(std::move(trie))
    , _max_token_delta(max_token_delta) {}
```
그리고 `apply()` 본문에서 `_trie.lookup(row, length, boost);` → `_trie->lookup(row, length, boost);` (포인터 역참조).

- [ ] **Step 5: 통과 확인**

Run: `cd /data/MyProject/stt/CTranslate2/build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test --gtest_filter='PhraseBiasTest.*' /tmp 2>&1 | tail -8`
Expected: 기존 PhraseBiasTest 전부 + `SharedTrieReusedByProcessors` PASS.

- [ ] **Step 6: 커밋**

```bash
cd /data/MyProject/stt/CTranslate2
git add include/ctranslate2/decoding_utils.h src/decoding_utils.cc tests/decoding_test.cc
git commit -m "feat(decoding): shareable PhraseBiasTrie via build_phrase_bias_trie + shared_ptr ctor

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: C++ — `WhisperOptions.compiled_phrase_bias_trie` + generate 사용

generate가 compiled trie 있으면 **rebuild 없이** 그걸로 processor 생성. 없고 raw `phrase_biases` 있으면 build(편의/C++ fallback).

**Files:** `include/ctranslate2/models/whisper.h`, `src/models/whisper.cc`

- [ ] **Step 1: 옵션 필드 추가** — `include/ctranslate2/models/whisper.h`의 `WhisperOptions`에서 기존 `std::vector<PhraseBias> phrase_biases;` 아래에:

```cpp
      // Pre-compiled trie (built once at load time). If set, used directly without
      // rebuilding per generate. Takes precedence over `phrase_biases`.
      std::shared_ptr<const PhraseBiasTrie> compiled_phrase_bias_trie;
```
(`PhraseBiasTrie`는 `ctranslate2` 네임스페이스 → `models` 안에서 `ctranslate2::PhraseBiasTrie`. whisper.h는 이미 `#include "ctranslate2/decoding_utils.h"` 함.)

- [ ] **Step 2: generate 주입부 교체** — `src/models/whisper.cc`의 기존 phrase bias 주입 블록을 아래로 교체:

```cpp
      std::shared_ptr<const PhraseBiasTrie> bias_trie = options.compiled_phrase_bias_trie;
      if (!bias_trie && !options.phrase_biases.empty())
        bias_trie = build_phrase_bias_trie(to_phrase_bias_entries(options.phrase_biases));
      if (bias_trie)
        decoding_options.logits_processors.emplace_back(
          std::make_shared<PhraseBiasProcessor>(bias_trie));
```
(`build_phrase_bias_trie`/`PhraseBiasProcessor`/`PhraseBiasTrie`는 `ctranslate2` 네임스페이스 — `models` 안에서 enclosing 네임스페이스 lookup으로 해소. 안 되면 `ctranslate2::` 명시.)

- [ ] **Step 3: 빌드 + 기존 C++ 회귀**

Run: `cd /data/MyProject/stt/CTranslate2/build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test ../tests/data 2>&1 | tail -6`
Expected: 빌드 성공, 기존 + PhraseBias 테스트 PASS, known-fail Gemm 3개 외 신규 실패 0.

- [ ] **Step 4: 커밋**

```bash
cd /data/MyProject/stt/CTranslate2
git add include/ctranslate2/models/whisper.h src/models/whisper.cc
git commit -m "feat(whisper): WhisperOptions.compiled_phrase_bias_trie (reuse, no per-generate rebuild)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Python 확장 빌드 환경 (prerequisite)

**Files:** (코드 변경 없음)

- [ ] **Step 1: libctranslate2 install**

Run:
```bash
cd /data/MyProject/stt/CTranslate2
cmake --install build --prefix "$PWD/install-cpu" 2>&1 | tail -3
ls install-cpu/lib*/libctranslate2.so* 2>/dev/null && echo "LIB OK"
```
Expected: `LIB OK`.

- [ ] **Step 2: install-* gitignore**

Run: `cd /data/MyProject/stt/CTranslate2 && git check-ignore install-cpu && echo IGNORED || echo ADD`
Expected: `IGNORED`. `ADD`면 `.gitignore`에 `/install-*` 추가 후 커밋.

- [ ] **Step 3: Python ext editable 설치**

Run:
```bash
cd /data/MyProject/stt/CTranslate2
export CTRANSLATE2_ROOT="$PWD/install-cpu"
export LD_LIBRARY_PATH="$CTRANSLATE2_ROOT/lib:$CTRANSLATE2_ROOT/lib64:$LD_LIBRARY_PATH"
pip install -e python/ 2>&1 | tail -5
```
Expected: `Successfully installed ctranslate2`.

- [ ] **Step 4: import smoke**

Run: `cd /data/MyProject/stt/CTranslate2 && LD_LIBRARY_PATH="$PWD/install-cpu/lib:$PWD/install-cpu/lib64:$LD_LIBRARY_PATH" python -c "import ctranslate2; print(ctranslate2.models.Whisper)"`
Expected: `<class 'ctranslate2._ext.Whisper'>`. (커밋 없음.)

---

## Task 4: Python — `PhraseBiasPath`/`PhraseBias` 타입 바인딩

엔진이 쓰는 필드만(`ids`, `step_bias`, `min_prefix_len`). `start_bias`/`mode` 미노출.

**Files:** `python/cpp/whisper.cc`, `python/ctranslate2/models/__init__.py`, `python/tests/test_phrase_bias.py`(신규)

- [ ] **Step 1: 실패 테스트** — `python/tests/test_phrase_bias.py` 신규:

```python
import ctranslate2


def test_phrase_bias_path_defaults():
    p = ctranslate2.models.PhraseBiasPath(ids=[10, 20, 30], step_bias=0.25)
    assert list(p.ids) == [10, 20, 30]
    assert abs(p.step_bias - 0.25) < 1e-6
    assert p.min_prefix_len == 1


def test_phrase_bias_path_min_prefix_len():
    p = ctranslate2.models.PhraseBiasPath(ids=[1, 2], step_bias=0.5, min_prefix_len=2)
    assert p.min_prefix_len == 2


def test_phrase_bias_wraps_paths():
    p1 = ctranslate2.models.PhraseBiasPath(ids=[100, 200, 300], step_bias=0.25)
    p2 = ctranslate2.models.PhraseBiasPath(ids=[101, 200, 300], step_bias=0.25)
    b = ctranslate2.models.PhraseBias(token_paths=[p1, p2])
    assert len(b.token_paths) == 2
    assert list(b.token_paths[0].ids) == [100, 200, 300]
```

- [ ] **Step 2: 실패 확인**

Run: `cd /data/MyProject/stt/CTranslate2 && LD_LIBRARY_PATH="$PWD/install-cpu/lib:$PWD/install-cpu/lib64:$LD_LIBRARY_PATH" python -m pytest python/tests/test_phrase_bias.py -q 2>&1 | tail -6`
Expected: FAIL — `module 'ctranslate2.models' has no attribute 'PhraseBiasPath'`.

- [ ] **Step 3: 타입 바인딩** — `python/cpp/whisper.cc`의 `void register_whisper(py::module& m) {` 다음, 첫 `py::class_<models::WhisperGenerationResult>` **앞**에:

```cpp
      py::class_<models::PhraseBiasPath>(
        m, "PhraseBiasPath",
        "One tokenization path of a phrase: token ids + precomputed continuation bias.")
        .def(py::init([](std::vector<size_t> ids, float step_bias, uint16_t min_prefix_len) {
               models::PhraseBiasPath p;
               p.ids = std::move(ids);
               p.step_bias = step_bias;
               p.min_prefix_len = min_prefix_len;
               return p;
             }),
             py::arg("ids"),
             py::kw_only(),
             py::arg("step_bias") = 0.f,
             py::arg("min_prefix_len") = 1,
             "Build a path from token ids and a precomputed per-step bias.")
        .def_readwrite("ids", &models::PhraseBiasPath::ids)
        .def_readwrite("step_bias", &models::PhraseBiasPath::step_bias)
        .def_readwrite("min_prefix_len", &models::PhraseBiasPath::min_prefix_len)
        .def("__repr__", [](const models::PhraseBiasPath& p) {
          return "PhraseBiasPath(ids=" + std::string(py::repr(py::cast(p.ids)))
            + ", step_bias=" + std::to_string(p.step_bias)
            + ", min_prefix_len=" + std::to_string(p.min_prefix_len) + ")";
        });

      py::class_<models::PhraseBias>(
        m, "PhraseBias",
        "A surface form compiled to one or more token paths (e.g. with/without leading space).")
        .def(py::init([](std::vector<models::PhraseBiasPath> token_paths) {
               models::PhraseBias b;
               b.token_paths = std::move(token_paths);
               return b;
             }),
             py::arg("token_paths"))
        .def_readwrite("token_paths", &models::PhraseBias::token_paths)
        .def("__repr__", [](const models::PhraseBias& b) {
          return "PhraseBias(token_paths=" + std::string(py::repr(py::cast(b.token_paths))) + ")";
        });

```

- [ ] **Step 4: re-export** — `python/ctranslate2/models/__init__.py`의 import 목록에 `PhraseBias`, `PhraseBiasPath` 추가 (알파벳 순):

```python
    from ctranslate2._ext import (
        PhraseBias,
        PhraseBiasPath,
        Wav2Vec2,
        Wav2Vec2Bert,
        Whisper,
        WhisperGenerationResult,
        WhisperGenerationResultAsync,
    )
```

- [ ] **Step 5: 재빌드 + 통과**

Run:
```bash
cd /data/MyProject/stt/CTranslate2
export CTRANSLATE2_ROOT="$PWD/install-cpu"; export LD_LIBRARY_PATH="$CTRANSLATE2_ROOT/lib:$CTRANSLATE2_ROOT/lib64:$LD_LIBRARY_PATH"
pip install -e python/ 2>&1 | tail -2
python -m pytest python/tests/test_phrase_bias.py -q 2>&1 | tail -6
```
Expected: 3 passed.

- [ ] **Step 6: 커밋**

```bash
cd /data/MyProject/stt/CTranslate2
git add python/cpp/whisper.cc python/ctranslate2/models/__init__.py python/tests/test_phrase_bias.py
git commit -m "feat(python): bind PhraseBiasPath and PhraseBias types

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Python — 생성자 `phrase_biases` 주입(persistent) + generate 사용

`WhisperWrapper`에 `_compiled_trie` 보관, 생성자에서 1회 build, generate가 per-call override 또는 `_compiled_trie`를 options에 주입.

**Files:** `python/cpp/whisper.cc`, `python/tests/test_phrase_bias.py`

- [ ] **Step 1: 실패 테스트(시그니처)** — `test_phrase_bias.py`에 추가:

```python
def test_whisper_ctor_and_generate_expose_phrase_biases():
    assert "phrase_biases" in (ctranslate2.models.Whisper.__init__.__doc__ or "")
    assert "phrase_biases" in (ctranslate2.models.Whisper.generate.__doc__ or "")
```

- [ ] **Step 2: 실패 확인**

Run: `cd /data/MyProject/stt/CTranslate2 && LD_LIBRARY_PATH="$PWD/install-cpu/lib:$PWD/install-cpu/lib64:$LD_LIBRARY_PATH" python -m pytest python/tests/test_phrase_bias.py::test_whisper_ctor_and_generate_expose_phrase_biases -q 2>&1 | tail -5`
Expected: FAIL.

- [ ] **Step 3: WhisperWrapper에 멤버 + 커스텀 생성자** — `python/cpp/whisper.cc`의 `class WhisperWrapper`에서 `using ReplicaPoolHelper::ReplicaPoolHelper;`를 **삭제**하고 커스텀 ctor + 멤버 추가:

```cpp
    class WhisperWrapper : public ReplicaPoolHelper<models::Whisper> {
    public:
      WhisperWrapper(const std::string& model_path,
                     const std::string& device,
                     const std::variant<int, std::vector<int>>& device_index,
                     const StringOrMap& compute_type,
                     size_t inter_threads,
                     size_t intra_threads,
                     long max_queued_batches,
                     bool flash_attention,
                     bool tensor_parallel,
                     py::object files,
                     const std::optional<std::vector<models::PhraseBias>>& phrase_biases)
        : ReplicaPoolHelper<models::Whisper>(model_path, device, device_index, compute_type,
                                             inter_threads, intra_threads, max_queued_batches,
                                             flash_attention, tensor_parallel, files) {
        if (phrase_biases && !phrase_biases->empty()) {
          auto entries = models::to_phrase_bias_entries(*phrase_biases);
          if (!entries.empty())
            _compiled_trie = build_phrase_bias_trie(entries);
        }
      }
```
그리고 클래스 `private:`에 멤버 추가:
```cpp
    private:
      std::shared_ptr<const PhraseBiasTrie> _compiled_trie;
```
(`build_phrase_bias_trie`/`PhraseBiasTrie`는 `ctranslate2::`. `<ctranslate2/models/whisper.h>`가 `decoding_utils.h`를 가져옴.)

- [ ] **Step 4: generate에 per-call + 멤버 주입** — `WhisperWrapper::generate` 시그니처 끝에 파라미터 추가:

```cpp
               float sampling_temperature,
               const std::optional<std::vector<models::PhraseBias>>& phrase_biases) {
```
본문에서 `options.suppress_blank = suppress_blank;` 다음에 (**3-way semantics**: None=model-level, []=disable, [...]=override):

```cpp
        if (phrase_biases) {                       // 명시적으로 전달됨 (빈 리스트 포함)
          if (!phrase_biases->empty())
            options.phrase_biases = *phrase_biases;  // [...] : per-call override (코어가 build)
          // [] : 이 호출만 disable — 아무것도 주입 안 함 (compiled trie도 X)
        } else {                                   // None : 생성자 model-level trie 사용 (없으면 nullptr=no-op)
          options.compiled_phrase_bias_trie = _compiled_trie;
        }
```

- [ ] **Step 5: py::init + generate arg 등록** — `python/cpp/whisper.cc`의 Whisper `py::class_` 에서 `.def(py::init<...>())`의 템플릿 인자 끝에 `const std::optional<std::vector<models::PhraseBias>>&` 추가하고 `py::arg("files")=py::none(),` 다음에 `py::arg("phrase_biases")=py::none(),` 추가, 그리고 init docstring `Arguments:` 끝에:
```
                   phrase_biases: Optional list of :class:`ctranslate2.models.PhraseBias`
                     (token ids + precomputed bias), compiled once into a reverse trie kept
                     for this model and reused across generate() calls. None disables biasing.
                     Keywords must be tokenized to ids by the caller (e.g. faster-whisper).
```
그리고 `.def("generate", ...)`의 `py::arg("sampling_temperature")=1,` 다음에:
```cpp
             py::arg("phrase_biases")=py::none(),
```
및 generate docstring `Arguments:` 끝에:
```
                   phrase_biases: Per-call phrase bias control.
                     None: use the model-level compiled trie set at construction (or no-op if none).
                     []: disable phrase bias for this call only (ablation switch).
                     [PhraseBias, ...]: override with these for this call (rebuilt per call;
                     intended for experiments/tests — production uses the constructor argument).
```

- [ ] **Step 6: 재빌드 + 통과**

Run:
```bash
cd /data/MyProject/stt/CTranslate2
export CTRANSLATE2_ROOT="$PWD/install-cpu"; export LD_LIBRARY_PATH="$CTRANSLATE2_ROOT/lib:$CTRANSLATE2_ROOT/lib64:$LD_LIBRARY_PATH"
pip install -e python/ 2>&1 | tail -2
python -m pytest python/tests/test_phrase_bias.py -q 2>&1 | tail -6
```
Expected: 4 passed.

- [ ] **Step 7: 커밋**

```bash
cd /data/MyProject/stt/CTranslate2
git add python/cpp/whisper.cc python/tests/test_phrase_bias.py
git commit -m "feat(python): Whisper(phrase_biases=...) ctor (persistent trie) + generate override

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: whisper-tiny 통합 (생성자/ per-call 효과 + no-op)

bias는 **생성된 토큰** suffix에 작용(step 0 미작용)하므로 **baseline 먼저** 뽑고 `g0` 다음 토큰을 강제.

> 의존: `transformers`+`torch`+네트워크. 없으면 skip.

**Files:** `python/tests/test_phrase_bias.py`

- [ ] **Step 1: 통합 테스트 작성** — 추가:

```python
import os

import numpy as np
import pytest

from . import test_utils


def _setup(tmp_dir):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    converter = ctranslate2.converters.TransformersConverter("openai/whisper-tiny")
    out = converter.convert(str(tmp_dir.join("ct2_whisper_tiny")))
    processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny")
    audio = np.load(os.path.join(test_utils.get_data_dir(), "audio", "jfk.npy"))
    feats = processor(audio, padding=False, sampling_rate=16000).input_features[0]
    feats = np.pad(feats, [(0, 0), (0, 3000 - feats.shape[-1])]).astype(np.float32)
    features = ctranslate2.StorageView.from_array(np.expand_dims(feats, 0))
    prompts = [["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]]
    return out, features, prompts


def _bias_on(g0, target):
    path = ctranslate2.models.PhraseBiasPath(ids=[g0, target], step_bias=50.0)
    return [ctranslate2.models.PhraseBias(token_paths=[path])]


@test_utils.only_on_linux
def test_percall_phrase_biases_noop_and_effect(tmp_dir):
    out, features, prompts = _setup(tmp_dir)
    model = ctranslate2.models.Whisper(out, device="cpu")
    base = model.generate(features, prompts, beam_size=1)[0].sequences_ids[0]
    assert len(base) >= 2
    # no-op: None / []
    assert model.generate(features, prompts, beam_size=1, phrase_biases=None)[0].sequences_ids[0] == base
    assert model.generate(features, prompts, beam_size=1, phrase_biases=[])[0].sequences_ids[0] == base
    # effect (per-call override)
    g0 = base[0]
    target = (base[1] + 1) % 1000
    if target == base[1]:
        target = (target + 1) % 1000
    biased = model.generate(features, prompts, beam_size=1,
                            phrase_biases=_bias_on(g0, target))[0].sequences_ids[0]
    assert biased[0] == g0
    assert biased[1] == target
    assert biased != base


@test_utils.only_on_linux
def test_constructor_phrase_biases_persistent(tmp_dir):
    out, features, prompts = _setup(tmp_dir)
    base = ctranslate2.models.Whisper(out, device="cpu").generate(
        features, prompts, beam_size=1)[0].sequences_ids[0]
    g0, b1 = base[0], base[1]
    target = (b1 + 1) % 1000
    if target == b1:
        target = (target + 1) % 1000
    # 생성자 주입 → 매 generate 재사용
    model = ctranslate2.models.Whisper(out, device="cpu", phrase_biases=_bias_on(g0, target))
    out1 = model.generate(features, prompts, beam_size=1)[0].sequences_ids[0]
    out2 = model.generate(features, prompts, beam_size=1)[0].sequences_ids[0]
    assert out1[1] == target and out2[1] == target   # 두 번 모두 동일 효과 (cached trie 재사용 — output stability)
    assert out1 == out2
    # None == model-level (biased)
    none_out = model.generate(features, prompts, beam_size=1, phrase_biases=None)[0].sequences_ids[0]
    assert none_out[1] == target
    # [] == 이 호출만 disable → baseline (ablation 스위치)
    disabled = model.generate(features, prompts, beam_size=1, phrase_biases=[])[0].sequences_ids[0]
    assert disabled == base
    assert disabled[1] == b1
```

> 재사용 주의: 위 `out1 == out2`는 **output stability**만 확인한다. "trie가 generate마다 rebuild 안 됨"은 구조(WhisperWrapper `_compiled_trie` 1회 build·read-only 참조)로 보장되며, 직접 증명(counter/hook)은 과해서 안 한다.

- [ ] **Step 2: 실행 (가능 환경)**

Run:
```bash
cd /data/MyProject/stt/CTranslate2
export LD_LIBRARY_PATH="$PWD/install-cpu/lib:$PWD/install-cpu/lib64:$LD_LIBRARY_PATH"
python -m pytest python/tests/test_phrase_bias.py -q 2>&1 | tail -15
```
Expected: 단위 4 + 통합 2 = 6 passed (transformers 없으면 통합 2 skip).

- [ ] **Step 3: 커밋**

```bash
cd /data/MyProject/stt/CTranslate2
git add python/tests/test_phrase_bias.py
git commit -m "test(python): whisper-tiny phrase_biases (ctor persistent + per-call + no-op)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: P3 완료 게이트

- [ ] **Step 1: C++ 게이트** — `cd /data/MyProject/stt/CTranslate2/build && ./tests/ctranslate2_test --gtest_filter='PhraseBiasTest.*' /tmp 2>&1 | tail -4` → 전부 PASS.
- [ ] **Step 2: Python 단위 게이트(필수)** — `LD_LIBRARY_PATH=... python -m pytest python/tests/test_phrase_bias.py -q 2>&1 | tail -6` → 단위 4 PASS(통합 PASS/skip).
- [ ] **Step 3: 기존 Python 회귀(스모크)** — `... python -m pytest python/tests/test_transformers.py -q -k whisper 2>&1 | tail -10` → 신규 실패 0(없으면 skip, collection 에러 0).
- [ ] **Step 4: STATUS 갱신 + 커밋**

```bash
cd /data/MyProject/stt/CTranslate2
git add dev-docs/STATUS.md
git commit -m "docs: mark Phase 3 complete in STATUS

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec 커버리지 (SSOT §5 P3 / §6 / 테스트 cat 6):**
- load-time persistent trie = Task 1(shared trie) + Task 2(option) + Task 5(생성자 1회 build·보관·주입). generate rebuild 없음 = Task 2 compiled trie 우선 + Task 5 멤버 주입.
- `PhraseBias*` 노출 = Task 4. 생성자 주입 = Task 5. per-call override = Task 5 generate.
- 3-way semantics(None=model-level / []=disable / [...]=override) = Task 5 Step 4 + Task 6 생성자 테스트(None→biased, []→baseline). 생성자 bias 없는 모델에선 None/[] 둘 다 no-op.
- empty no-op(§7-5) = compiled_trie nullptr(생성자 미주입) 또는 [] disable → processor 안 생성.
- 효과(cat 6 a/b/c/d) = Task 6 두 테스트(생성자 persistent 재사용 + per-call override + None/[] semantics).
- 토크나이저 없음(§0.1) = Python API는 ids만. tokenizer correctness 없음(P4).
- ReplicaPool race 없음 = trie는 `shared_ptr<const>` immutable, 생성자 고정, replica는 읽기만.

**Placeholder 스캔:** 모든 step 실제 코드/명령. "적절히 처리" 류 없음.

**타입 일관성:** `build_phrase_bias_trie(entries)→shared_ptr<const PhraseBiasTrie>` · `PhraseBiasProcessor(shared_ptr<const PhraseBiasTrie>)` · `WhisperOptions.compiled_phrase_bias_trie` · `WhisperWrapper._compiled_trie` · Python `PhraseBiasPath(ids,step_bias,min_prefix_len)`/`PhraseBias(token_paths)` — Task 1~6 동일.

**주의(실행자):**
- Task 5에서 `using ReplicaPoolHelper::ReplicaPoolHelper;` 삭제 후 커스텀 ctor로 교체 — base ctor 인자 순서/타입은 `python/cpp/replica_pool.h`의 `ReplicaPoolHelper(...)`와 정확히 일치시켜야 함(model_path, device, device_index, compute_type, inter_threads, intra_threads, max_queued_batches, flash_attention, tensor_parallel, files).
- `.def(py::init<...>())` 템플릿 인자 목록 끝에 `const std::optional<std::vector<models::PhraseBias>>&` 추가, `py::arg` 순서도 맞출 것.
- 모든 python 실행에 `LD_LIBRARY_PATH`(install-cpu/lib[64]).
- 효과 테스트 `target`이 baseline g1과 같으면 보정. step_bias=50은 fp32 logit을 압도하려는 값.

## 다음 (P3 이후)
- **P4** faster-whisper 연동 — 키워드→2 path **tokenizer compile**(pure 함수) + `Whisper(phrase_biases=...)` 생성자 연동 + 실제 음성 A/B. faster-whisper repo, 별도 plan.
