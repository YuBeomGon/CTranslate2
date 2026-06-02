# Phase 2 Implementation Plan — GPU `indexed_add` + CPU/GPU Parity

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (또는 subagent-driven-development). 체크박스(`- [ ]`)로 추적, 각 Task 끝에 커밋.
> 결정·금지사항·exemplar는 `dev-docs/SSOT.md`가 기준. 이 플랜은 **P2만**의 실행 지시서. P1(CPU)은 완료됨(`e849bea6`).

**Goal:** P1에서 만든 `PhraseBiasProcessor` 경로를 GPU에서도 동작시킨다 — `indexed_add` primitive에 CUDA 구현을 추가하고(인터페이스 불변), CPU/GPU 결과 parity를 fp32 타이트 + fp16/bf16 tolerance로 검증한다.

**Architecture:** P1 processor는 이미 `DEVICE_AND_TYPE_DISPATCH` 경유로 `primitives<D>::indexed_add`를 호출한다. P2는 **`src/cuda/primitives.cu`에 CUDA 커널 1개 + launch + 인스턴스화만 추가** → processor/trie 코드는 한 줄도 안 바뀌고 GPU에서 동작하게 된다. 커널은 `penalize_previous_tokens_kernel` 패턴을 그대로 차용(half는 float 경유, unique index라 race 없음). 검증은 primitive 레벨(device-parameterized) + processor 레벨(FloatType-parameterized)의 기존 테스트 인프라를 재사용.

**Tech Stack:** CUDA 12.6, Thrust/hand-written kernel, Google Test, CMake. RTX 4080 SUPER(sm_89) 확인됨.

### P2 범위 (고정)
- **포함:** CUDA `indexed_add` 커널+launch+인스턴스화 · CUDA 빌드 디렉터리 셋업 · primitive parity 테스트(CPU/CUDA fp32, device-parameterized) · processor parity 테스트(CPU fp32 + CUDA fp32/fp16/bf16, **batch>1**, **overlap 합산**, **shared prefix**) · tolerance 검증(`expect_storage_eq(error)`).
- **제외:** 실제 모델 generate 통합 parity(beam end-to-end, 통합 테스트 카테고리 6) · Python binding(P3) · tokenizer(P3) · faster-whisper(P4) · 성능 벤치(별도 §12).

### 금지사항 (SSOT §7, P2 관련)
- **새 CUDA 스타일 만들기 금지** — `indexed_fill`(`primitives.cu:63`) / `penalize_previous_tokens_kernel`(`primitives.cu:284`) 패턴을 **그대로** 따른다. 외부 CUDA 튜토리얼 보지 말 것.
- **`indexed_add`에 중복 index 금지** — CUDA에서 같은 index에 여러 thread `+=` = race. dedupe/합산은 processor(CPU)가 호출 전에 끝낸다(P1에서 `std::map`으로 이미 보장). 커널은 unique index 가정.
- 인터페이스 불변 — 시그니처는 P1과 동일: `static void indexed_add(T* x, const T* deltas, const int32_t* indices, dim_t num_indices);`

### GPU portability requirement (특정 GPU/CUDA 버전에 묶이면 안 됨)
이 기능은 **특정 NVIDIA 아키텍처/CUDA 버전 전용이면 안 된다.** 로컬 검증 GPU(RTX 4080 SUPER, sm_89)는 **smoke test일 뿐 release 기준이 아니다.**

- **소스는 arch-중립** — `#if __CUDA_ARCH__ >= xxx`, `sm_89` hardcode, 특정 intrinsic **금지**. 커널은 `x[idx] += delta`라 최신 API 불필요.
- **CMake arch flag 직접 추가 금지** — CT2가 관리하는 `CUDA_ARCH_LIST` 메커니즘만 사용. `src/cuda/primitives.cu`(= 기존 primitive와 같은 compilation unit)에 넣으면 CT2의 arch 설정을 그대로 탄다.
  - ⚠️ **핵심(코드 확인됨, `CMakeLists.txt:530`):** `CUDA_ARCH_LIST` 기본값 `Auto`는 `cuda_select_nvcc_arch_flags`가 **로컬 GPU 하나(sm_89)만** 감지 → **그 GPU 전용 바이너리**. 소스가 portable해도 바이너리는 아님.
  - **로컬 smoke 빌드:** `Auto` (빠름, 단일 arch) — Task 1.
  - **portable/release 빌드:** **`-DCUDA_ARCH_LIST=Common`** (멀티 arch fat binary, `CUDA_COMMON_GPU_ARCHITECTURES`) — Task 1 Step 5에서 컴파일 호환만 확인.
- **CUDA 버전 의존 API 금지** — cooperative groups, 최신 atomic 등 X. 기존 helper만: `cuda::get_cuda_stream()`, `cuda::device_cast()`, `DEVICE_AND_TYPE_DISPATCH`.
- **atomic 금지** — unique index(§7)라 `atomicAdd` 불필요.
- **dtype parity 정책(기존 `OpDeviceFPTest` 매트릭스를 그대로 따름):** fp32 필수 · fp16 필수 · **bf16도 동일하게 테스트**(CT2가 이미 CUDA bf16를 무조건 인스턴스화 → 우리만 skip하면 스위트와 불일치). 런타임에 bf16 미지원 GPU(pre-Ampere) fallback은 **CT2 상위 레이어의 기존 책임**이지 본 단위 테스트가 새로 게이트하지 않는다.

### 빌드/테스트
- **CPU 빌드(기존, P1 회귀용):** `build/` — `cd build && make -j"$(nproc)" ctranslate2_test`
- **CUDA 빌드(P2 신규):** `build-cuda/` (Task 1에서 셋업)
- **게이트 방식(무시 대신 분리 기록):** 1차 게이트는 **`--gtest_filter`로 우리 테스트 + 관련 CUDA primitive만** 돌려 전부 PASS 확인. 전체 회귀는 참고용이며, **known-fail = `CPU/OpDeviceFPTest.{Gemm,GemmBias,GemmResidual}/float32` 3개(Ruy 아티팩트, P2 무관)**. 판정 기준은 "이 3개 외 **신규 실패 0**". known-fail 목록은 이 줄이 SSOT.
- exemplar 라인(SSOT §11): 선언 `primitives.h:22`(P1에서 추가됨) · CPU `src/cpu/primitives.cc` · CUDA fill `src/cuda/primitives.cu:63` · 커널 패턴 `:284`(launch :311) · CUDA 인스턴스화 매크로 `DECLARE_IMPL` `:754`(`indexed_fill` `:762`).

---

## File Structure

| 파일 | 책임 | 변경 |
|------|------|------|
| `src/cuda/primitives.cu` | `indexed_add` CUDA 커널 + launch + 인스턴스화 | Task 2 |
| `tests/primitives_test.cc` | primitive parity (device-parameterized fp32) — P1 plain 테스트 대체 | Task 2 |
| `tests/decoding_test.cc` | processor parity (FloatType-parameterized) | Task 3 |
| (빌드) `build-cuda/` | CUDA 빌드 디렉터리 | Task 1 |

> 주의: P1 코드(`decoding_utils.*`, `whisper.*`, `primitives.h`, `src/cpu/primitives.cc`)는 **수정 없음**. P2는 CUDA 구현과 테스트만 추가한다.

---

## Task 1: CUDA 빌드 디렉터리 셋업 (prerequisite)

P2 테스트를 실제로 돌리려면 `WITH_CUDA=ON` 빌드가 필요하다. cuDNN은 불필요(`WITH_CUDNN=OFF`). 우리 커널을 추가하기 **전에** 기존 CUDA 경로가 이 머신에서 빌드·실행되는지 먼저 확인한다(셋업 문제와 우리 코드 문제를 분리).

**Files:** (코드 변경 없음 — 빌드 구성만)

- [ ] **Step 1: CUDA 빌드 구성**

Run:
```bash
cd /data/MyProject/stt/CTranslate2
cmake -S . -B build-cuda \
  -DBUILD_TESTS=ON -DBUILD_CLI=OFF \
  -DWITH_MKL=OFF -DWITH_RUY=ON \
  -DWITH_CUDA=ON -DWITH_CUDNN=OFF \
  -DOPENMP_RUNTIME=NONE -DCMAKE_BUILD_TYPE=Release
```
Expected: configure 성공, 로그에 `-DCT2_WITH_CUDA` 및 detected CUDA arch(예: `sm_89`) 표시. 에러 없이 `build-cuda/` 생성.

- [ ] **Step 2: 테스트 바이너리 빌드 (시간 소요)**

Run: `cmake --build build-cuda -j"$(nproc)" --target ctranslate2_test`
Expected: 링크까지 성공(`Built target ctranslate2_test`). CUDA 컴파일이라 수~수십 분 소요될 수 있음. 경고는 무방, **에러 0**.

- [ ] **Step 3: 기존 CUDA primitive 경로 확인 (우리 코드 추가 전 sanity)**

Run: `cd /data/MyProject/stt/CTranslate2 && ./build-cuda/tests/ctranslate2_test --gtest_filter='CUDA/PrimitiveTest.PenalizePreviousTokens' tests/data`
Expected: `[  PASSED  ] 1 test.` (GPU 런타임·디스패치 정상). 만약 0 tests run이면 CUDA 인스턴스화가 안 된 것 — 구성 재확인.

- [ ] **Step 4: 커밋 없음 (빌드 산출물)**

`build-cuda/`는 커밋하지 않는다. `.gitignore`에 `build*/`가 이미 잡히는지 확인:

Run: `cd /data/MyProject/stt/CTranslate2 && git check-ignore build-cuda && echo IGNORED || echo "NOT IGNORED — add to .gitignore"`
Expected: `IGNORED`. 만약 `NOT IGNORED`면 `.gitignore`에 `build-cuda/` 한 줄 추가하고 그 변경만 커밋: `git add .gitignore && git commit -m "chore: ignore build-cuda/"`.

- [ ] **Step 5: portable 멀티-arch 컴파일 호환 확인 (release 빌드 검증, 1회)**

로컬 `Auto` 빌드는 sm_89 단일 arch라 portability를 보장하지 못한다. **멀티-arch(`Common`) 구성이 컴파일되는지**만 별도 디렉터리에서 확인(실행은 로컬 GPU 한정이라 컴파일 성공이 게이트):

Run:
```bash
cd /data/MyProject/stt/CTranslate2
cmake -S . -B build-cuda-portable \
  -DBUILD_TESTS=ON -DBUILD_CLI=OFF -DWITH_MKL=OFF -DWITH_RUY=ON \
  -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DOPENMP_RUNTIME=NONE \
  -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH_LIST=Common
cmake --build build-cuda-portable -j"$(nproc)" --target ctranslate2 2>&1 | tail -5
```
Expected: 라이브러리 타깃 `ctranslate2`가 **여러 `-gencode arch=compute_xx` flag로 에러 없이 컴파일**(`Built target ctranslate2`). 이는 Task 2 커널 추가 **후 한 번 더** 돌려 우리 커널이 모든 common arch에서 컴파일되는지 확인하는 게이트(아래 Task 2 Step 6에서 재실행). `build-cuda-portable/`도 커밋하지 않음.

---

## Task 2: CUDA `indexed_add` 커널 + primitive parity 테스트

`penalize_previous_tokens_kernel`(`primitives.cu:284`) 패턴으로 커널 작성. half 타입은 float 경유(P1 CPU 구현과 동일 의미). P1의 plain `IndexedAddTest.CPU`를 **device-parameterized `TEST_P`로 대체**해 CPU/CUDA fp32 parity를 한 테스트로 본다.

**Files:**
- Modify: `src/cuda/primitives.cu` (커널 `:283` 근처, launch, 인스턴스화 `:762`)
- Modify: `tests/primitives_test.cc` (P1 `IndexedAddTest` 대체)

- [ ] **Step 1: 실패 테스트로 교체** — `tests/primitives_test.cc`의 P1 plain 테스트를 device-parameterized로 교체.

기존 (P1, `975e26f8`에서 추가):
```cpp
TEST(IndexedAddTest, CPU) {
  std::vector<float> x = {0, 1, 2, 3, 4};
  std::vector<float> deltas = {0.5f, 0.25f};
  std::vector<int32_t> indices = {1, 3};
  primitives<Device::CPU>::indexed_add(x.data(), deltas.data(), indices.data(), 2);
  std::vector<float> expected = {0, 1.5f, 2, 3.25f, 4};
  EXPECT_EQ(x, expected);
}
```
를 아래로 **교체** (`INSTANTIATE_TEST_SUITE_P(CPU, PrimitiveTest, ...)` 위, 다른 `TEST_P(PrimitiveTest, ...)`들과 나란히):
```cpp
TEST_P(PrimitiveTest, IndexedAdd) {
  const Device device = GetParam();
  StorageView x({5}, std::vector<float>{0, 1, 2, 3, 4}, device);
  StorageView deltas({2}, std::vector<float>{0.5f, 0.25f}, device);
  StorageView indices({2}, std::vector<int32_t>{1, 3}, device);
  StorageView expected({5}, std::vector<float>{0, 1.5f, 2, 3.25f, 4}, device);
  DEVICE_DISPATCH(device, primitives<D>::indexed_add(x.data<float>(),
                                                     deltas.data<float>(),
                                                     indices.data<int32_t>(),
                                                     indices.size()));
  expect_storage_eq(x, expected);
}
```
(`PrimitiveTest`는 `TestWithParam<Device>`이고 CPU+CUDA로 이미 인스턴스화되어 있어, 이 한 테스트가 두 디바이스에서 자동 실행된다. 그래서 P1의 별도 `IndexedAddTest` suite는 더 이상 불필요.)

- [ ] **Step 2: CUDA 빌드해서 실패 확인**

Run: `cmake --build build-cuda -j"$(nproc)" --target ctranslate2_test`
Expected: 링크 실패 — `undefined reference to ... primitives<Device::CUDA>::indexed_add(...)`. (CPU는 P1에서 구현됨, CUDA만 없음.)

> **Implementation note (실행자 필독):** 아래 커널/launch 코드는 **예시**다. float16/bfloat16 변환·stream 처리·device cast·block/grid 설정·template 인스턴스화는 **기존 CT2 CUDA primitive 스타일을 그대로 복사**한다. 새 cast/launch 패턴을 만들지 말 것. 예시 코드가 그대로 컴파일 안 되면 `static_cast<float>`를 고집하지 말고 `penalize_previous_tokens_kernel`(`:284`)이 half를 다루는 방식(중간 `float` 변수 경유, 암시적 변환)을 따른다.

- [ ] **Step 3: CUDA 커널 추가** — `src/cuda/primitives.cu`의 `indexed_fill` CUDA 구현(`:63`) **바로 위**에 (launch보다 파일상 앞에 와야 함):

`penalize_previous_tokens_kernel`이 half를 다루는 방식(`const float score = previous_scores[i];` 로 읽고, float 연산 후 `scores[idx] = ...` 로 다시 씀 — 암시적 device_type↔float 변환)을 **그대로** 따른다:
```cpp
  template <typename T>
  __global__ void indexed_add_kernel(T* x,
                                     const T* deltas,
                                     const int32_t* indices,
                                     cuda::index_t num_indices) {
    for (cuda::index_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_indices;
         i += blockDim.x * gridDim.x) {
      const cuda::index_t idx = indices[i];
      // unique index 계약(SSOT §7-7)이라 동일 idx 동시 쓰기 없음 → race 없음 → atomic 불필요.
      // half/bf16는 penalize_previous_tokens_kernel과 동일하게 float 경유.
      const float updated = static_cast<float>(x[idx]) + static_cast<float>(deltas[i]);
      x[idx] = updated;
    }
  }
```
(만약 `static_cast<float>(x[idx])`가 device_type에서 컴파일 안 되면, penalize처럼 `const float cur = x[idx];` 암시적 변환으로 바꾼다. 새 변환 헬퍼 만들지 말 것.)

- [ ] **Step 4: launch 함수 추가** — `src/cuda/primitives.cu`의 `indexed_fill`(`:63`) CUDA 구현 **바로 아래**에 (Step 3 커널이 그 위에 정의돼 있어 순서 OK):

```cpp
  template<>
  template <typename T>
  void primitives<Device::CUDA>::indexed_add(T* x, const T* deltas,
                                             const int32_t* indices, dim_t num_indices) {
    if (num_indices == 0)
      return;
    dim3 block(32);  // penalize_previous_tokens launch(:309)와 동일. <100개 scatter라 perf 무관 — 일관성 위해 32 유지(128/256으로 바꾸지 말 것).
    dim3 grid((num_indices + block.x - 1) / block.x);
    indexed_add_kernel<<<grid, block, 0, cuda::get_cuda_stream()>>>(
      cuda::device_cast(x),
      cuda::device_cast(deltas),
      indices,
      num_indices);
  }
```
(`block`/`grid`/`device_cast`/`get_cuda_stream`은 `penalize_previous_tokens`(`:309~318`)와 동일 형식. 새 launch 패턴 만들지 말 것.)

- [ ] **Step 5: 인스턴스화 추가** — `src/cuda/primitives.cu`의 `DECLARE_IMPL`(`:754`) 매크로 안, `indexed_fill` 인스턴스화 줄(`:762`) **바로 아래**(백슬래시 줄맞춤 유지):

```cpp
  template void                                                         \
  primitives<Device::CUDA>::indexed_add(T*, const T*, const int32_t*, dim_t); \
```

- [ ] **Step 6: CUDA 빌드 + 통과 확인 (+ portable arch 컴파일 게이트)**

Run: `cmake --build build-cuda -j"$(nproc)" --target ctranslate2_test && ./build-cuda/tests/ctranslate2_test --gtest_filter='*PrimitiveTest.IndexedAdd/*' tests/data`
Expected: 2 PASS — `CPU/PrimitiveTest.IndexedAdd`, `CUDA/PrimitiveTest.IndexedAdd`.

그리고 **우리 커널이 모든 common arch에서 컴파일되는지**(portability) 재확인:
Run: `cmake --build build-cuda-portable -j"$(nproc)" --target ctranslate2 2>&1 | tail -5`
Expected: `Built target ctranslate2` (여러 `-gencode` flag로 에러 없이). 특정 arch에서만 컴파일되는 코드가 들어가면 여기서 실패.

- [ ] **Step 7: CPU 빌드 회귀 확인** (CPU 빌드에선 device 파라미터가 CPU만 → 1 PASS, 깨지지 않았는지)

Run: `cd /data/MyProject/stt/CTranslate2/build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test --gtest_filter='*PrimitiveTest.IndexedAdd/*' /tmp`
Expected: 1 PASS — `CPU/PrimitiveTest.IndexedAdd`. (P1 plain 테스트 제거로 인한 누락 없음.)

- [ ] **Step 8: 커밋**

```bash
cd /data/MyProject/stt/CTranslate2
git add src/cuda/primitives.cu tests/primitives_test.cc
git commit -m "feat(primitives): add indexed_add (CUDA) + device-parameterized parity test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `PhraseBiasProcessor` CPU/GPU FP parity 테스트

processor 코드는 **변경 없음**(P1의 `DEVICE_AND_TYPE_DISPATCH` 경로가 Task 2 커널로 GPU에서 동작). 여기서는 `FloatType`-parameterized 테스트로 CPU fp32 + CUDA fp32/fp16/bf16 parity를 본다. **row-wise batch>1**, **shared prefix**, **overlap 합산**을 한 케이스에 담는다.

> **범위 명확화:** batch=2는 `PhraseBiasProcessor`의 **row-wise 동작 + batch>1**을 커버한다. logits row 관점에서 beam은 batch 차원에 흡수되지만, 이 테스트는 **beam search의 reorder/gather/prefix divergence end-to-end parity를 보장하지 않는다** — 그건 범위 밖(실제 모델 통합 테스트, 카테고리 6). "beam 1/5 통과"라고 읽지 말 것.

**Files:**
- Modify: `tests/decoding_test.cc` (기존 PhraseBiasTest 아래 + 파일 끝 인스턴스화)

- [ ] **Step 1: parity 테스트 추가** — `tests/decoding_test.cc`의 마지막 PhraseBias 테스트(`ConvertModelOptionToEntries`) 아래에 fixture + 테스트 추가. (이건 "먼저 실패해야 하는" 테스트가 아니다 — Task 2 커널이 끝난 상태이므로 작성 즉시 통과해야 한다. CUDA 케이스는 Task 2에 의존.):

```cpp
// CPU/GPU parity: logits는 device/dtype, sequences는 host(int32) — 실제 generate 계약과 동일.
class PhraseBiasProcessorFPTest : public ::testing::TestWithParam<FloatType> {
};

TEST_P(PhraseBiasProcessorFPTest, CpuGpuParity) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;

  // batch=2 (row-wise batch>1 커버; beam end-to-end 아님), vocab=6, logits 모두 0.
  StorageView logits({2, 6}, std::vector<float>(12, 0.f), device);
  logits = logits.to(dtype);
  DisableTokens disable(logits, std::numeric_limits<float>::lowest());

  // shared prefix [1,2,*] 두 path → overlap 합산 검증.
  std::vector<PhraseBiasEntry> entries = {
    {{1, 2, 3}, 0.3f, 1},
    {{1, 2, 4}, 0.3f, 1},
  };
  PhraseBiasProcessor proc(entries);

  // sequences는 항상 host(CPU). row0 suffix [...,1] → token2 boost(0.3+0.3=0.6).
  //                              row1 suffix [1,2]  → token3,4 boost(각 0.3).
  StorageView seq({2, 2}, std::vector<int32_t>{0, 1,  1, 2});
  proc.apply(2, logits, disable, seq, {0}, nullptr);

  StorageView expected({2, 6}, std::vector<float>{
      0, 0, 0.6f, 0,    0,    0,    // row0: token2 += 0.6 (합산)
      0, 0, 0,    0.3f, 0.3f, 0},   // row1: token3,4 += 0.3
      Device::CPU);
  expect_storage_eq(logits, expected.to(device).to(dtype), error);
}

INSTANTIATE_TEST_SUITE_P(CPU, PhraseBiasProcessorFPTest,
                         ::testing::Values(FloatType{Device::CPU, DataType::FLOAT32, 1e-5}),
                         fp_test_name);
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, PhraseBiasProcessorFPTest,
                         ::testing::Values(FloatType{Device::CUDA, DataType::FLOAT32, 1e-5},
                                           FloatType{Device::CUDA, DataType::FLOAT16, 1e-2},
                                           FloatType{Device::CUDA, DataType::BFLOAT16, 4e-2}),
                         fp_test_name);
#endif
```

> 주의(CPU dtype, **코드 확인됨** `src/decoding_utils.cc` DisableTokens 생성자): CPU일 때만 `_logits_data = logits.data<float>()`를 잡고(fp16 CPU면 dtype assert), **CUDA면 `_logits_data = nullptr`**(`data<float>()` 미호출) + 나머지 멤버는 dim metadata뿐 → **CUDA fp16/bf16 생성 안전**. 따라서 **CPU param은 fp32만**, CUDA는 fp16/bf16 포함.

- [ ] **Step 2: CUDA 빌드해서 실패 확인** (아직 fixture만 있고 통과 기준 미검증 상태 — 컴파일은 되고 CUDA 케이스가 실제로 도는지 확인)

Run: `cmake --build build-cuda -j"$(nproc)" --target ctranslate2_test`
Expected: 컴파일·링크 성공. (이 테스트는 Task 2 커널에 의존하므로 Task 2 완료 후에만 통과 가능.)

- [ ] **Step 3: CUDA parity 통과 확인**

Run: `cd /data/MyProject/stt/CTranslate2 && ./build-cuda/tests/ctranslate2_test --gtest_filter='*PhraseBiasProcessorFPTest.*' tests/data`
Expected: 4 PASS — `CPU/...FLOAT32`, `CUDA/...FLOAT32`, `CUDA/...FLOAT16`, `CUDA/...BFLOAT16`.

- [ ] **Step 4: CPU 빌드 회귀 확인**

Run: `cd /data/MyProject/stt/CTranslate2/build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test --gtest_filter='PhraseBiasTest.*:*PhraseBiasProcessorFPTest.*' /tmp`
Expected: P1의 모든 PhraseBiasTest + `CPU/PhraseBiasProcessorFPTest.FLOAT32` PASS.

- [ ] **Step 5: 커밋**

```bash
cd /data/MyProject/stt/CTranslate2
git add tests/decoding_test.cc
git commit -m "test(decoding): add CPU/GPU parity for PhraseBiasProcessor (fp32 tight, fp16/bf16 tol)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: P2 완료 게이트 — 우리 테스트 1차 게이트 + 전체 회귀(참고)

- [ ] **Step 1: 1차 게이트 — 우리 테스트 + 관련 CUDA primitive만 필터 (반드시 전부 PASS)**

Run: `cd /data/MyProject/stt/CTranslate2 && ./build-cuda/tests/ctranslate2_test --gtest_filter='*PrimitiveTest.IndexedAdd:PhraseBiasTest.*:*PhraseBiasProcessorFPTest.*:*PrimitiveTest.PenalizePreviousTokens' tests/data`
Expected: 전부 PASS (CPU+CUDA IndexedAdd, P1 PhraseBiasTest 전체, parity 4종, 기존 CUDA primitive sanity). **이 게이트가 P2 합격 기준.**

- [ ] **Step 2: 전체 회귀 (참고용 — 신규 실패 0 확인)**

Run: `cd /data/MyProject/stt/CTranslate2 && ./build-cuda/tests/ctranslate2_test tests/data 2>&1 | tail -20`
그리고 CPU 빌드도: `cd /data/MyProject/stt/CTranslate2/build && ./tests/ctranslate2_test ../tests/data 2>&1 | tail -12`
Expected: **known-fail 3개(`CPU/OpDeviceFPTest.{Gemm,GemmBias,GemmResidual}/float32`, Ruy 아티팩트) 외 신규 실패 0.** P1 대비 PASS 수가 새 parity 케이스만큼 늘어남. known-fail 외 실패가 1개라도 새로 생기면 게이트 불합격 — 원인 추적.

- [ ] **Step 3: STATUS 갱신 + 커밋** — `dev-docs/STATUS.md`의 Phase 2 섹션을 `✅ 완료`로, 각 task에 commit SHA 기록 후:

```bash
cd /data/MyProject/stt/CTranslate2
git add dev-docs/STATUS.md
git commit -m "docs: mark Phase 2 complete in STATUS

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec 커버리지 (SSOT §6 P2 / 테스트 카테고리 5 대조):**
- GPU sparse `indexed_add`, 기존 primitive 패턴 재사용 = Task 2 (`penalize_previous_tokens_kernel` 차용, 새 CUDA 금지 준수).
- CPU/GPU parity, fp32 타이트 = Task 2(primitive) + Task 3(processor, `error=1e-5`).
- fp16 tolerance(allclose) = Task 3 `FLOAT16 1e-2`, bf16 `4e-2` (`expect_storage_eq(error)` = `EXPECT_NEAR`).
- batch>1 = Task 3 **batch=2 (row-wise)**. ⚠️ beam reorder/gather end-to-end parity는 **범위 외**(통합 카테고리 6) — "beam 1/5 통과"로 읽지 말 것(Step 1 범위 명확화 박스).
- overlap 합산 일치 = Task 3 shared prefix `[1,2,3]/[1,2,4]` → token2 += 0.6.
- unique index 계약(§7-7) = 커널 주석 + processor(P1) dedupe 유지. 커널은 합산 안 함(중복 없다고 가정), **atomic 미사용**.
- **GPU portability** = arch-중립 소스 + `CUDA_ARCH_LIST=Common` 멀티-arch 컴파일 게이트(Task 1 Step 5 / Task 2 Step 6). `Auto`는 로컬 smoke 전용. bf16는 기존 `OpDeviceFPTest` 매트릭스 그대로.
- 인터페이스 불변 = 시그니처 P1과 동일, processor 코드 무변경.

**Placeholder 스캔:** 모든 코드 step에 실제 코드/명령. "적절히 처리" 류 없음. 커널 정의-launch 순서 주의(Step 4)와 CPU dtype 제약(Task 3 주의)을 명시.

**타입 일관성:** `indexed_add(T*, const T*, const int32_t*, dim_t)` — P1 선언/CPU와 동일, CUDA launch·인스턴스화·테스트 전부 일치. `FloatType{device,dtype,error}`·`fp_test_name`·`expect_storage_eq(error)` = `test_utils.h`/`ops_test.cc` 기존 그대로. `PhraseBiasEntry{ids,step_bias,min_prefix_len}`·`PhraseBiasProcessor(entries)` = P1과 동일.

**주의(실행자):**
- Task 1 CUDA 빌드는 **수~수십 분**(`Common` 멀티-arch는 더 김). `build-cuda/`·`build-cuda-portable/`는 커밋 금지.
- Step 3/4(Task 2): `indexed_add_kernel` 정의가 launch 함수보다 **파일에서 앞**에 있어야 컴파일됨 — 커널을 `indexed_fill` CUDA 구현 위에 둔다.
- **예시 커널 코드는 illustrative.** half/bf16 cast·stream·device cast·block/grid·인스턴스화는 기존 `penalize_previous_tokens_kernel`/`indexed_fill` 스타일을 **그대로 복사**. `static_cast<float>`가 안 되면 penalize의 암시적 변환으로 교체. 새 cast/launch 패턴 만들지 말 것.
- `cuda::device_cast` / `cuda::index_t` / `cuda::get_cuda_stream` 는 `penalize_previous_tokens`(`:284`,`:311`)에서 그대로 쓰는 것 — 새로 만들지 말 것.
- **block(32) 유지** — penalize와 동일, 128/256으로 바꾸지 말 것(코드베이스 일관성, perf 무관).
- CPU parity param은 fp32만(DisableTokens dtype 제약, 코드 확인됨). CUDA는 fp16/bf16 포함.
- **arch hardcode/atomic/cooperative groups/버전 의존 API 금지** (GPU portability requirement 섹션).

## 실행 노트 (실제와 차이 — 2026-06-02 실행, ✅ P2 완료)
> 플랜 본문은 "의도" 기록으로 보존. 실제 실행에서 아래가 달랐다(재독 시 참고). 재사용 교훈은 SSOT §13/§11에 반영됨.

1. **Task 1 baseline 빌드 순서** — 플랜은 "커널 추가 전 baseline CUDA 빌드 성공"을 가정했으나, **P1 processor가 이미 `DEVICE_AND_TYPE_DISPATCH`로 `primitives<Device::CUDA>::indexed_add`를 참조**해 우리 커널 없이는 **link 실패**. 이 link 에러 자체가 Task 2 Step 2의 기대 실패와 동일 → Task 2 커널을 먼저 구현 후 sanity(PenalizePreviousTokens) 확인. (교훈: SSOT §13 "레이어링 사실".)
2. **파라미터화 테스트 필터** — `--gtest_filter='*PrimitiveTest.IndexedAdd'`는 0 매칭. `TEST_P`는 param suffix `/0`이 붙어 **`/*` 필요**: `'*PrimitiveTest.IndexedAdd/*'`. (본문 필터 인라인 수정함.)
3. **`--gtest_list_tests`도 positional data-dir 인자 필수** (없으면 `missing data directory` throw).

**최종 결과:** primitive parity(CPU+CUDA fp32) 2 PASS · processor parity(CPU fp32 + CUDA fp32/fp16/bf16) 4 PASS · 1차 게이트 17 PASS · 전체 회귀 CUDA 363 passed(known-fail Gemm 3개 외 신규 0) · portable `Common` 멀티-arch(sm_53~86 SASS + compute_86 PTX) 컴파일 OK. 커널 코드는 `static_cast<float>` 그대로 half/bf16 컴파일됨(fallback 불필요). 커밋 T2 `38a5b4c0` · T3 `424a7450`.

## 다음 (P2 이후)
- **P3** CT2 Python 바인딩만 (`PhraseBias*` 노출 + `generate(phrase_biases=...)`, **ids+bias in, 토크나이저 없음** — SSOT §0.1) — 별도 plan
- **P4** faster-whisper 연동 — **tokenizer compile**(키워드→2 path, special 제거·leading-space·roundtrip) + 실제 A/B. faster-whisper repo — 별도 plan
