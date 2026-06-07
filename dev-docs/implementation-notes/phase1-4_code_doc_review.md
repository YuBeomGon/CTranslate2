# CT2 KenLM Fusion Phase 1-4 Code and Doc Review

Review date: 2026-06-07

Reviewed commits:

- `67fa8de2 feat: add LM fusion phase 1 scaffolding`
- `459ea4e2 feat: implement LM fusion beam selection core`
- `8203969c fix: allocate LM fusion topk buffers on logits device`
- `48c3c632 feat: add Whisper LM fusion API wiring`
- `4475d108 feat: add KenLM BPE fusion scorer`

정본 설계는 [`../ct2_kenlm_fusion_design.md`](../ct2_kenlm_fusion_design.md)이고,
구현 중 변경/확정된 사항은 [`phase1-4_design_deltas.md`](phase1-4_design_deltas.md)에 분리했다.

## Findings

### Blocker: none

Phase 1-4 구현에서 즉시 rollback해야 할 차단 이슈는 확인하지 못했다.
기본 build, KenLM OFF error path, KenLM ON link/smoke path는 검증됐다.

### Medium: hard-prefix LM state sync 전용 테스트 부족

- 관련 코드: `src/decoding.cc:148`, `src/decoding.cc:749`, `src/decoding.cc:782`, `src/decoding.cc:802`
- 현재 정책: hard-prefix forced/update step에서는 fusion scoring을 우회하고, `update_sample_with_prefix()` 이후 최종 token/gather index 기준으로 LM state를 advance한다.
- 위험: 이 경로는 상태 drift가 발생해도 transcript만 보면 바로 드러나지 않을 수 있다.
- 현재 테스트: `tests/decoding_test.cc`의 FakeScorer 테스트는 top-k strict/no-rescue 중심이다.
- 권장 후속 작업: hard prefix가 beam origin을 덮는 케이스에서 FakeScorer state history까지 검증하는 unit test를 추가한다.

### Medium: Whisper prompt replay API-level test 부족

- 관련 코드: `src/models/whisper.cc:274`, `src/models/whisper.cc:319`, `src/kenlm_fusion.cc:68`
- 현재 정책: forwarded prompt 중 `token_id < _eot_id` text token만 initial history로 전달하고 scorer가 replay한다.
- 위험: previous-text prompt, language/task token, timestamp-enabled path와 실제 faster-whisper 운영 조합에서 parity가 깨질 수 있다.
- 권장 후속 작업: Whisper generate API smoke에서 prompt가 있는 경우와 없는 경우의 scorer initial history 차이를 검증한다.

### Medium: KenLM fixture test가 score semantics를 충분히 고정하지 않음

- 관련 코드: `tests/kenlm_fusion_test.cc:24`
- 현재 테스트: `CT2_KENLM_TEST_BINARY`가 있을 때 scorer 생성과 finite score만 확인한다.
- 위험: `t<ID>` word lookup, initial history replay 순서, log10 to ln 변환이 회귀해도 finite score만으로는 놓칠 수 있다.
- 권장 후속 작업: 작은 ARPA/binary fixture 생성 정책을 정하고, 특정 transition score가 기대 ln 값과 맞는지 검증한다.

### Low: `KENLM_MAX_ORDER`는 외부 KenLM build와 수동으로 맞춰야 함

- 관련 코드: `CMakeLists.txt:29`, `CMakeLists.txt:296`
- 현재 정책: `KENLM_MAX_ORDER` cache value를 CT2 compile definition으로 전달한다.
- 위험: 링크되는 KenLM library와 다른 값으로 빌드하면 ABI/state layout이 맞지 않을 수 있다.
- 권장 후속 작업: `KENLM_ROOT/build/CMakeCache.txt`가 있으면 `KENLM_MAX_ORDER` 값을 읽어 mismatch를 조기에 fail하는 CMake guard를 추가한다.

### Low: path-keyed KenLM shared cache 미구현

- 관련 코드: `src/models/whisper.cc:325`, `src/kenlm_fusion.cc:185`
- 현재 정책: `lm_fusion_model_path`가 주어질 때마다 scorer를 생성한다.
- 영향: 기능 문제는 아니지만 반복 호출 시 KenLM binary load 비용이 커질 수 있다.
- 권장 후속 작업: canonical path 기반 shared cache를 v1.1로 추가한다.

### Low: Python extension smoke 미검증

- 관련 코드: `python/cpp/whisper.cc:50`, `python/cpp/whisper.cc:271`
- 현재 상태: C++ core와 tests target은 검증했지만 Python extension build/smoke는 별도 실행하지 않았다.
- 권장 후속 작업: wheel 또는 editable extension build 후 `Whisper.generate(..., lm_fusion_*)` kwargs binding smoke를 실행한다.

## Resolved During Review

- 설계 문서의 `LmFusionScorer::make_initial_states` 예시가 구현 시그니처와 달랐다.
- 조치: `initial_histories`, `beam_size` 인자를 포함하도록 [`../ct2_kenlm_fusion_design.md`](../ct2_kenlm_fusion_design.md)를 갱신했다.
- 구현 플랜의 `DecodingOptions`/`BeamSearch` 예시도 `lm_initial_histories` 보관 및 scorer replay 위임 정책을 반영하도록 갱신했다.

## Verification Evidence

Phase 4 완료 시점에 확인한 명령:

```bash
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_CLI=OFF -DWITH_MKL=OFF -DWITH_DNNL=OFF -DWITH_OPENBLAS=OFF -DWITH_CUDA=OFF -DWITH_RUY=ON -DWITH_KENLM=OFF
cmake --build build --target ctranslate2_test -j2
./build/tests/ctranslate2_test tests/data --gtest_filter=DecodingTest.LmFusion*:KenlmFusionTest.*:ModelTest.UpdateDecoderOutputLayer
```

```bash
cmake -S . -B build-kenlm -DBUILD_TESTS=ON -DBUILD_CLI=OFF -DWITH_MKL=OFF -DWITH_DNNL=OFF -DWITH_OPENBLAS=OFF -DWITH_CUDA=OFF -DWITH_RUY=ON -DWITH_KENLM=ON -DKENLM_ROOT=/tmp/kenlm
cmake --build build-kenlm --target ctranslate2_test -j2
CT2_KENLM_TEST_BINARY=/tmp/ct2_lm_fusion_test.binary LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./build-kenlm/tests/ctranslate2_test tests/data --gtest_filter=KenlmFusionTest.*:DecodingTest.LmFusion*
```

```bash
./build/tests/ctranslate2_test tests/data --gtest_filter=-CPU/OpDeviceFPTest.Gemm*
```

결과:

- `WITH_KENLM=OFF` 관련 테스트 통과
- `WITH_KENLM=ON` link 및 fixture smoke 통과
- Gemm FP 계열 기존 Ruy backend 차이 항목을 제외한 전체 기본 테스트 통과

## Recommended Next Order

1. hard-prefix LM state sync unit test
2. deterministic KenLM score fixture test
3. Whisper Python/API smoke
4. path-keyed KenLM shared cache
5. 실제 운영 KenLM binary 기준 WER/latency benchmark
