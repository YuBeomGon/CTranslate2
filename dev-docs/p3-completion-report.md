# P3 Completion Report — CT2 Python Binding + Load-time Persistent Trie

> 기준 문서: `dev-docs/SSOT.md`, `dev-docs/p3-impl-plan.md`
> 작성일: 2026-06-04

## 완료 요약

P3 구현은 완료됐다. CT2 Python 바인딩에서 `PhraseBiasPath`/`PhraseBias`를 노출하고,
`Whisper(..., phrase_biases=...)` 생성자 주입과
`generate(..., phrase_biases=None/[]/[...])` 호출별 override를 지원한다.

생성자에 전달된 phrase biases는 `WhisperWrapper`에서 trie로 1회 build되어
`shared_ptr<const PhraseBiasTrie>`로 보관되고, generate 호출 시
`WhisperOptions.compiled_phrase_bias_trie`로 주입된다. 따라서 상용 경로인
생성자 주입은 generate마다 trie를 rebuild하지 않는다.

## 커밋

- `92a7cb89` `feat(decoding): share phrase bias trie`
- `a4d0f651` `feat(whisper): reuse compiled phrase bias trie`
- `aa023f71` `chore: ignore local install prefixes`
- `ab4134eb` `chore: ignore editable extension artifact`
- `cab9ba65` `feat(python): bind whisper phrase bias types`
- `e91699c0` `feat(python): add whisper phrase_biases injection`
- `15b0aadb` `test(python): cover whisper phrase bias integration`
- `1376ffeb` `docs(p3): mark phase 3 complete`

## 검증 결과

- C++ PhraseBias gate:
  - `./tests/ctranslate2_test --gtest_filter=PhraseBiasTest.* /tmp`
  - 결과: `10 passed`
- Python phrase bias:
  - `python -m pytest python/tests/test_phrase_bias.py -q`
  - 결과: `6 passed`
- 기존 Whisper smoke:
  - `python -m pytest python/tests/test_transformers.py -q -k whisper`
  - 결과: `1 passed, 8 skipped, 46 deselected`
- C++ 전체 회귀:
  - `./tests/ctranslate2_test ../tests/data`
  - 결과: `196 passed, 1 skipped`
  - 실패: 기존 known-fail `CPU/OpDeviceFPTest.{Gemm,GemmBias,GemmResidual}/float32` 3개만 실패

## 계획 대비 조정 사항

1. Python editable 설치가 현재 conda/base Python 환경에 적용됐다.
   - `pip install -e python/` 실행으로 기존 `ctranslate2 4.6.0`이 제거되고
     이 repo의 editable `ctranslate2 4.7.2`가 설치됐다.
   - 별도 conda env는 일반적으로 영향받지 않지만, 같은 env를 쓰는 다른 작업은 영향받을 수 있다.

2. Python import 시 런타임 라이브러리 경로 문제가 있었다.
   - `_ext`의 Linux RPATH가 `/usr/local/lib64:/usr/local/lib`를 먼저 보게 되어
     `/usr/local/lib/libctranslate2.so.4`를 잡았다.
   - 이 환경의 conda `libstdc++.so.6`에는 `GLIBCXX_3.4.32`가 없어 import가 실패했다.
   - 검증 명령은 아래처럼 system `libstdc++`와 local `libctranslate2`를 `LD_PRELOAD`로 지정해 실행했다.

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6:/data/MyProject/stt/CTranslate2/install-cpu/lib/libctranslate2.so.4
LD_LIBRARY_PATH=/data/MyProject/stt/CTranslate2/install-cpu/lib:/data/MyProject/stt/CTranslate2/install-cpu/lib64:$LD_LIBRARY_PATH
```

3. T6 whisper-tiny 통합 테스트의 target 선택 방식을 조정했다.
   - 원래 계획은 baseline의 초반 토큰을 기준으로 target을 강제하는 단순 형태였다.
   - 실제 whisper-tiny logits에서는 두 번째 token margin이 `PhraseBiasProcessor`의 최종 clamp보다 커서 테스트가 skip될 수 있었다.
   - 그래서 baseline `return_logits_vocab` 전체 step을 훑고, 현재 기본 `+2.0` clamp로 실제 argmax를 뒤집을 수 있는 step을 자동 선택하도록 바꿨다.
   - 구현 계약인 최종 delta clamp는 그대로 유지했고, 테스트만 특정 step에 덜 의존하게 조정했다.

4. T6 audio fixture 경로 확인 방식을 조정했다.
   - `test_utils.get_data_dir()`는 audio만 필요한 테스트에서도 transliteration model fixture까지 요구해 skip을 유발했다.
   - P3 통합 테스트는 `tests/data/audio/jfk.npy`만 필요하므로 해당 파일만 직접 확인하도록 바꿨다.

5. whisper-tiny 테스트는 네트워크/cache 접근이 필요했다.
   - `transformers`/Hugging Face가 캐시 확인과 모델 파일 접근을 수행하므로 sandbox network 제한에서는 실패하거나 오래 재시도했다.
   - 최종 검증은 네트워크 허용으로 실행했다.

## 남은 범위

P3는 CT2 Python binding까지만 다룬다. 키워드 문자열을 Whisper tokenizer로
token ids + step bias로 compile하는 일은 P4 faster-whisper 범위다.
