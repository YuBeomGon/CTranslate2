# P4 Completion Report — faster-whisper Phrase Bias

> 기준 문서: `dev-docs/SSOT.md`, `dev-docs/p4-impl-plan.md`
> 작성일: 2026-06-04 · 구현 repo: `/data/MyProject/stt/faster-whisper` 브랜치 `feature/phrase-bias`

## 완료 요약

faster-whisper에 사용자용 phrase bias config(JSON/dict)를 추가했다. `WhisperModel(..., phrase_bias_config=...)`로
도메인 키워드를 받아 **모델 init 1회**에 자기 토크나이저로 token id path로 compile하고, P3의
`ctranslate2.models.Whisper(phrase_biases=...)` 생성자(persistent trie)로 넘긴다. 런타임 `transcribe()`는
재토크나이즈·재compile하지 않는다. CT2는 여전히 ids+step_bias만 받고 문자열을 받지 않는다(SSOT §0.1 경계 유지).

핵심 모듈 `faster_whisper/phrase_bias.py`가 SSOT "제일 중요" tokenizer correctness의 실제 실행 위치다:
각 surface를 `" "+surface`/`surface` 2 variant로 인코딩 → special 제거(`id < eot_id`) → roundtrip 검증
(`decode(ids)==variant`) → `len<2` skip → uniform/ramp schedule로 step_bias 분배 → CT2 DTO 변환.
ramp는 누적 증분을 `min_prefix_len` 1·2·3 엔트리로 표현해 **CT2의 overlap 합산이 자동으로 ramp를 구현**한다(CT2 변경 없음).

## faster-whisper 커밋 (`feature/phrase-bias`)

- `a065e52` `test: add phrase bias compiler tests`
- `b9076d3` `feat: compile phrase bias config`
- `1a810c0` `feat: pass phrase bias config to CTranslate2`
- `1b94fb4` `test: verify phrase bias with whisper tokenizer`
- `8fa28c2` `docs: add phrase bias config usage`
- `7ade840` `test: add phrase bias A/B runner`

## CTranslate2 문서 커밋

- (이 커밋) `docs(p4): mark faster-whisper integration complete` — `dev-docs/STATUS.md` Phase 4 ✅, 본 보고서.
- 별도: `dev-docs/p1-p3-audit-report.md` (서브에이전트 P1–P3 감사: 버그 없음, P4 진행 가능).

## 검증 결과

모든 Python 실행은 P3 빌드 링크 때문에 아래 prefix 필요(아래 "조정 사항 5" 참조):
`LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6:.../install-cpu/lib/libctranslate2.so.4`,
`LD_LIBRARY_PATH=.../install-cpu/lib(:lib64)`.

- faster-whisper 집중 테스트:
  - `tests/test_phrase_bias.py` → **11 passed** (config 파싱, uniform/ramp, special 제거, roundtrip, 1-token skip, enabled=false, CT2 DTO, init handoff, 충돌, 실제 whisper tiny 토크나이저 2 path)
  - `tests/test_tokenizer.py` → **3 passed** (실제 `WhisperModel("tiny"/"tiny.en")` 생성 = 재구조화한 `__init__` end-to-end)
  - `tests/test_transcribe.py::test_transcribe_signature` `::test_hotwords` → **2 passed** (signature parity 유지, 기존 hotwords 동작 무변경)
- CT2 P3 호환 스모크:
  - `python/tests/test_phrase_bias.py` → **6 passed**
- A/B 러너 스모크:
  - `benchmark/phrase_bias_ab.py --model tiny --manifest <hotwords> --phrase-bias-config examples/phrase_bias_config.json --device cpu` → exit 0, JSONL rows + summary 출력. (tiny 모델이라 이 샘플에선 term flip 없음 — 러너 동작 스모크이지 도메인 acceptance 아님.)

## 계획 대비 조정 사항

1. **ramp 테스트 float 비교** — 플랜 테스트는 dataclass 정확 일치(`==`)로 계산된 float를 비교했는데
   `0.2 - 0.1 = 0.09999999999999999 != 0.1`로 실패. 구현(컴파일러)은 정상이라 **테스트를** `pytest.approx`(step_bias)로
   수정(ids/min_prefix_len은 정확 비교). 컴파일러 출력은 안 바꿈.
2. **충돌 검사 위치** — 플랜은 `phrase_bias_config` + raw `phrase_biases` 충돌 검사를 토크나이저 로드 **뒤**에 뒀으나,
   오설정 시 불필요한 토크나이저 다운로드(네트워크)를 막으려 **토크나이저 로드 앞**으로 옮김.
3. **토크나이저 fallback** — phrase_biases를 CT2 생성자에 넘기려면 모델 생성 **전에** 토크나이저가 필요해
   `__init__`에서 토크나이저 로딩을 모델 생성 앞으로 옮겼다. 기존 fallback은 `self.model.is_multilingual`(모델 필요)을
   썼는데, 모델 전이라 쓸 수 없어 `_fallback_tokenizer_name`(모델명 `.en`/`-en` suffix 기반)으로 대체.
   tokenizer.json이 모델 디렉터리에 있으면 fallback은 안 타므로 일반 경로 영향 없음.
4. **A/B 러너 import 경로** — `python benchmark/phrase_bias_ab.py`를 그냥 실행하면 **site-packages의 (구버전)
   faster_whisper**가 shadow해 `phrase_bias_config`가 CT2로 그대로 흘러가 `TypeError`. repo를 우선시키려면
   `PYTHONPATH=/data/MyProject/stt/faster-whisper` 필요(또는 repo를 editable 설치). pytest는 repo 루트가
   sys.path에 있어 로컬을 써서 영향 없었음. → 운영/배포 시 faster-whisper editable 설치 권장.
5. **런타임 라이브러리 경로** — `import ctranslate2`가 conda `libstdc++`(GLIBCXX_3.4.32 없음) 때문에 실패.
   P3 보고서와 동일하게 system `libstdc++` + local `libctranslate2`를 `LD_PRELOAD`/`LD_LIBRARY_PATH`로 지정해 실행.

## 남은 범위

- 실제 도메인 음성으로의 본격 A/B(recall/precision/insertion/CER·WER/latency 정량) — 도메인 manifest·오디오 필요.
  현재는 러너 + hotwords 스모크까지. (SSOT 테스트 cat 7의 정식 acceptance는 도메인 데이터 확보 후.)
- ramp schedule의 insertion 위험은 도메인별 A/B로 검증 필요(README에 경고 명시).
- (보류) negative/suppress(block) 모드.
