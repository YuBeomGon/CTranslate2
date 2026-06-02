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

### 빌드/테스트
- **CPU 빌드(기존, P1 회귀용):** `build/` — `cd build && make -j"$(nproc)" ctranslate2_test`
- **CUDA 빌드(P2 신규):** `build-cuda/` (Task 1에서 셋업)
- **무시할 baseline 실패:** `CPU/OpDeviceFPTest.Gemm/GemmBias/GemmResidual /float32` (Ruy 아티팩트). 항상 필터로 본인 테스트만 확인.
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

Run: `cd /data/MyProject/stt/CTranslate2 && git status --short --ignored build-cuda 2>/dev/null | head -1; git check-ignore build-cuda && echo IGNORED || echo "NOT IGNORED — add to .gitignore"`
Expected: `IGNORED`. 만약 `NOT IGNORED`면 `.gitignore`에 `build-cuda/` 한 줄 추가하고 그 변경만 커밋: `git add .gitignore && git commit -m "chore: ignore build-cuda/"`.

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

- [ ] **Step 3: CUDA 커널 추가** — `src/cuda/primitives.cu`의 `penalize_previous_tokens_kernel` 정의 **바로 위**(약 `:283`)에:

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
      // unique index 계약(SSOT §7-7)이라 동일 idx에 동시 쓰기 없음 → race 없음.
      // half 타입은 float 경유(CPU 구현과 동일 의미).
      x[idx] = static_cast<float>(x[idx]) + static_cast<float>(deltas[i]);
    }
  }
```

- [ ] **Step 4: launch 함수 추가** — `src/cuda/primitives.cu`의 `indexed_fill`(`:63`) CUDA 구현 **바로 아래**에:

```cpp
  template<>
  template <typename T>
  void primitives<Device::CUDA>::indexed_add(T* x, const T* deltas,
                                             const int32_t* indices, dim_t num_indices) {
    if (num_indices == 0)
      return;
    dim3 block(32);
    dim3 grid((num_indices + block.x - 1) / block.x);
    indexed_add_kernel<<<grid, block, 0, cuda::get_cuda_stream()>>>(
      cuda::device_cast(x),
      cuda::device_cast(deltas),
      indices,
      num_indices);
  }
```
(`indexed_add_kernel`이 Step 3에서 `penalize_previous_tokens_kernel` 위에 정의되어 이 launch보다 먼저 보이도록 — 즉 커널 정의가 파일에서 launch보다 위에 있어야 한다. Step 3 위치(`:283` 근처)는 launch(`:63`)보다 아래이므로, **커널 정의를 launch보다 앞**(예: `indexed_fill` 구현 위, 파일 상단 커널 모음 근처)에 두거나, 이 launch 함수를 커널 정의 아래로 옮긴다. 가장 단순: 커널 정의를 `indexed_fill` CUDA 구현 **바로 위**에 두고, launch를 그 아래에 둔다.)

- [ ] **Step 5: 인스턴스화 추가** — `src/cuda/primitives.cu`의 `DECLARE_IMPL`(`:754`) 매크로 안, `indexed_fill` 인스턴스화 줄(`:762`) **바로 아래**(백슬래시 줄맞춤 유지):

```cpp
  template void                                                         \
  primitives<Device::CUDA>::indexed_add(T*, const T*, const int32_t*, dim_t); \
```

- [ ] **Step 6: CUDA 빌드 + 통과 확인**

Run: `cmake --build build-cuda -j"$(nproc)" --target ctranslate2_test && ./build-cuda/tests/ctranslate2_test --gtest_filter='*PrimitiveTest.IndexedAdd' tests/data`
Expected: 2 PASS — `CPU/PrimitiveTest.IndexedAdd`, `CUDA/PrimitiveTest.IndexedAdd`.

- [ ] **Step 7: CPU 빌드 회귀 확인** (CPU 빌드에선 device 파라미터가 CPU만 → 1 PASS, 깨지지 않았는지)

Run: `cd /data/MyProject/stt/CTranslate2/build && make -j"$(nproc)" ctranslate2_test && ./tests/ctranslate2_test --gtest_filter='*PrimitiveTest.IndexedAdd' /tmp`
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

processor 코드는 **변경 없음**(P1의 `DEVICE_AND_TYPE_DISPATCH` 경로가 Task 2 커널로 GPU에서 동작). 여기서는 `FloatType`-parameterized 테스트로 CPU fp32 + CUDA fp32/fp16/bf16 parity를 본다. **batch>1**(beam 확장과 동치), **shared prefix**, **overlap 합산**을 한 케이스에 담는다.

**Files:**
- Modify: `tests/decoding_test.cc` (기존 PhraseBiasTest 아래 + 파일 끝 인스턴스화)

- [ ] **Step 1: 실패 테스트 작성** — `tests/decoding_test.cc`의 마지막 PhraseBias 테스트(`ConvertModelOptionToEntries`) 아래에 fixture + 테스트 추가:

```cpp
// CPU/GPU parity: logits는 device/dtype, sequences는 host(int32) — 실제 generate 계약과 동일.
class PhraseBiasProcessorFPTest : public ::testing::TestWithParam<FloatType> {
};

TEST_P(PhraseBiasProcessorFPTest, CpuGpuParity) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;

  // batch=2 (beam>1/batch>1 동치), vocab=6, logits 모두 0.
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

> 주의(CPU dtype): `DisableTokens` 생성자는 CPU일 때 `logits.data<float>()`를 잡으므로 **CPU param은 fp32만** 둔다(fp16 CPU면 dtype assert). CUDA는 `data<float>()`를 호출하지 않아 fp16/bf16 안전.

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

## Task 4: P2 완료 게이트 — 전체 회귀 (CPU + CUDA)

- [ ] **Step 1: CUDA 빌드 전체 테스트**

Run: `cd /data/MyProject/stt/CTranslate2 && ./build-cuda/tests/ctranslate2_test tests/data 2>&1 | tail -20`
Expected: 우리 테스트(`*IndexedAdd`, `PhraseBiasTest.*`, `*PhraseBiasProcessorFPTest.*`) 전부 PASS. CUDA 쪽 기존 테스트 PASS. (CPU baseline Gemm 3개 실패는 무시.)

- [ ] **Step 2: CPU 빌드 전체 테스트**

Run: `cd /data/MyProject/stt/CTranslate2/build && ./tests/ctranslate2_test ../tests/data 2>&1 | tail -12`
Expected: P1과 동일하게 194 passed 류(+ 새 parity CPU 케이스). baseline Gemm 3개 외 실패 없음.

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
- beam 1/5 = logits 레벨에선 batch 차원에 흡수됨 → Task 3 **batch=2**로 커버(주석 명시). end-to-end beam parity는 범위 외(통합 카테고리 6, 별도).
- batch>1 = Task 3 batch=2.
- overlap 합산 일치 = Task 3 shared prefix `[1,2,3]/[1,2,4]` → token2 += 0.6.
- unique index 계약(§7-7) = 커널 주석 + processor(P1) dedupe 유지. 커널은 합산 안 함(중복 없다고 가정).
- 인터페이스 불변 = 시그니처 P1과 동일, processor 코드 무변경.

**Placeholder 스캔:** 모든 코드 step에 실제 코드/명령. "적절히 처리" 류 없음. 커널 정의-launch 순서 주의(Step 4)와 CPU dtype 제약(Task 3 주의)을 명시.

**타입 일관성:** `indexed_add(T*, const T*, const int32_t*, dim_t)` — P1 선언/CPU와 동일, CUDA launch·인스턴스화·테스트 전부 일치. `FloatType{device,dtype,error}`·`fp_test_name`·`expect_storage_eq(error)` = `test_utils.h`/`ops_test.cc` 기존 그대로. `PhraseBiasEntry{ids,step_bias,min_prefix_len}`·`PhraseBiasProcessor(entries)` = P1과 동일.

**주의(실행자):**
- Task 1 CUDA 빌드는 **수~수십 분**. `build-cuda/`는 커밋 금지.
- Step 3/4(Task 2): `indexed_add_kernel` 정의가 launch 함수보다 **파일에서 앞**에 있어야 컴파일됨 — 커널을 `indexed_fill` CUDA 구현 위에 두는 걸 권장.
- `cuda::device_cast` / `cuda::index_t` / `cuda::get_cuda_stream` 는 `penalize_previous_tokens`(`:284`,`:311`)에서 그대로 쓰는 것 — 새로 만들지 말 것.
- CPU parity param은 fp32만(DisableTokens dtype 제약).

## 다음 (P2 이후)
- **P3** Python binding + tokenizer compile (special 제거·leading-space 유지·space/no-space path) — 별도 plan
- **P4** faster-whisper 연동 — 별도 plan
