# Phrase Bias 테스트 매뉴얼 (A/B 검증)

> **이 문서는 구현 완료 *이후*의 경험적 검증(post-implementation verification)을 위한 매뉴얼입니다.**
> 즉, 포크에 `WhisperModel(..., phrase_bias_config=...)` / `phrase_biases=[...]` API가 **이미 구현되어 동작하는 상태**를 전제로 합니다. API가 아직 없다면 이 매뉴얼이 아니라 [`SSOT.md`](SSOT.md) §6 로드맵을 먼저 따르세요.
>
> **설계 출처(반드시 교차 참조)**: [[SSOT.md]]
> - 검증 지표 정의 → **[`SSOT.md`](SSOT.md) §5 가드레일/검증 규칙**
> - YAML / bias 데이터 모델 → **[`SSOT.md`](SSOT.md) §4 데이터 모델**
> - 빌드 흐름·평가 지표 근거 → [`deep-research-report.md`](deep-research-report.md) §성능과 빌드 전략, §검증과 운영 로드맵
>
> **상태**: 검증용 매뉴얼 (last updated: 2026-06-01)

---

## 목표 한 줄 요약

같은 Whisper 모델 · 같은 오디오 · 같은 디코딩 파라미터에서 **phrase bias만 켜고 끈 두 빌드**를 비교해, 도메인 용어 인식이 **품질을 해치지 않으면서** 개선되는지 숫자로 증명한다.

| | Version A (baseline) | Version B (fork) |
|---|---|---|
| CT2 wheel | upstream `pip install ctranslate2` | 로컬 빌드 phrase-bias wheel (`X.Y.Z+pb1`) |
| phrase bias | **OFF** | **ON** (도메인 vocab) |
| 모델 / 오디오 / 디코딩 파라미터 | 동일 | 동일 |

유일한 독립 변수는 **bias 적용 여부**다. 나머지는 전부 고정한다.

---

## 1. 목적 & 검증 가설

### 가설 (H1)
> phrase bias **ON** 시 **domain term recall ↑**, 단 **insertion ratio는 거의 그대로**(또는 무시할 수준 증가)이고, **대표 오인식(misrecognition)은 줄어든다.**

### 합격 기준 (pass / fail criteria)

A → B로 갈 때, 동일 테스트셋에서 아래를 **모두** 만족하면 **PASS**:

| # | 지표 | 합격 조건 | 근거 |
|---|------|-----------|------|
| C1 | domain term recall | `recall_B - recall_A >= +0.05` (절대 +5%p 이상) | 핵심 효과 |
| C2 | misrecognition rate | `misrec_B <= misrec_A` (대표 오인식이 줄거나 동일) | negative bias 효과 |
| C3 | insertion ratio | `ins_B - ins_A <= +0.01` (절대 +1%p 이하) | 과편향(over-biasing) 방지 |
| C4 | overall WER | `WER_B - WER_A <= +0.005` (전체 품질 비악화) | 부작용 가드 |
| C5 | domain term precision | `prec_B >= prec_A - 0.02` (정밀도 급락 금지) | 헛삽입 방지 |

> C1만 통과하고 C3/C4/C5 중 하나라도 실패하면 **FAIL (over-biasing 의심)**. recall과 insertion ratio는 [`SSOT.md`](SSOT.md) §5 가드레일대로 **항상 함께** 본다.

임계값(`0.05`, `0.01` 등)은 도메인/데이터 규모에 맞춰 §1 표를 직접 조정하되, **사전에 고정**하고 결과를 본 뒤 바꾸지 않는다(p-hacking 방지).

---

## 2. 디렉토리 구조

테스트 벤치는 저장소 루트 옆 또는 별도 작업 폴더에 둔다. 예시 경로: `/data/MyProject/stt/CTranslate2/tests-phrase-bias/`

```
tests-phrase-bias/
  audio/                 # 실제 wav/mp3 (도메인 용어가 실제 발화된 파일)
    sample_001.wav
    sample_002.wav
    ...
  refs/                  # 정답 전사 (audio 파일별 .txt, 같은 stem)
    sample_001.txt
    sample_002.txt
    ...
  vocab/
    domain_terms.yaml    # 도메인 키워드 (positive/negative). SSOT §4 스키마
  models/
    large-v3-ct2/        # 변환된 Whisper 모델 (A/B 공통, 단 하나만 둔다)
  results/
    A/                   # Version A 출력 전사 (.txt) + meta
      sample_001.txt
      ...
    B/                   # Version B 출력 전사 (.txt) + meta
      sample_001.txt
      ...
    scores.csv           # score.py 산출 per-file 점수
    summary.json         # A vs B 집계 + pass/fail
  run_ab.py              # 모델 로드 → 전사 → results/<A|B>/ 덤프
  score.py              # results/ 채점 → scores.csv / summary.json
  env.lock.txt           # 재현성: 버전·해시·하드웨어 기록 (§8)
```

> `models/`에는 변환 모델을 **하나만** 둔다. A와 B가 *반드시 같은* 모델 디렉토리를 가리켜야 한다(§3, §9 참조).

---

## 3. 두 버전 환경 구성

두 빌드를 **분리된 venv**로 격리한다. 같은 인터프리터에 두 wheel을 번갈아 설치하면 캐시·잔존 `.so` 때문에 오염되므로 금지.

### 3.0 공통 — Whisper 모델 변환 (1회)

A/B가 공유할 CT2 모델을 미리 만든다. 이미 변환된 모델이 있으면 건너뛴다.

```bash
# 변환 전용 임시 venv (어느 쪽에서 해도 무방, 결과물만 공유)
python -m venv /tmp/venv-convert
source /tmp/venv-convert/bin/activate
pip install -U "ctranslate2" "transformers[torch]"

ct2-transformers-converter \
  --model openai/whisper-large-v3 \
  --output_dir /data/MyProject/stt/CTranslate2/tests-phrase-bias/models/large-v3-ct2 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16
deactivate
```

> 변환 모델의 해시를 기록해 둔다(§8). 같은 디렉토리를 A/B 둘 다 가리킨다.

### 3.1 Version A — upstream baseline

```bash
python -m venv /data/MyProject/stt/CTranslate2/tests-phrase-bias/.venv-A
source /data/MyProject/stt/CTranslate2/tests-phrase-bias/.venv-A/bin/activate

pip install -U pip
pip install "ctranslate2" "faster-whisper" pyyaml jiwer

# 기록: 실제로 설치된 upstream 버전 고정 확인
python -c "import ctranslate2; print('CT2(A)=', ctranslate2.__version__)"
deactivate
```

Version A는 phrase bias 기능이 없는 vanilla 빌드다. `run_ab.py`는 A 모드에서 bias 인자를 **아예 전달하지 않는다**(§5).

### 3.2 Version B — fork phrase-bias wheel 빌드

[`deep-research-report.md`](deep-research-report.md) §성능과 빌드 전략의 빌드 흐름과 [`CLAUDE.md`](../CLAUDE.md) 빌드 명령을 그대로 사용한다.

```bash
# 0) 포크 체크아웃 (이미 이 저장소가 포크라면 생략)
#    feature/whisper-phrase-bias 브랜치인지 확인
cd /data/MyProject/stt/CTranslate2
git rev-parse --abbrev-ref HEAD     # feature/whisper-phrase-bias 기대

# 1) C++ 라이브러리 빌드 (CUDA 환경 기준)
mkdir -p build && cd build
cmake .. \
  -DBUILD_TESTS=ON \
  -DENABLE_PROFILING=OFF \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON
make -j"$(nproc)"
sudo make install
sudo ldconfig

# (선택) C++ 단위 테스트로 코어가 정상인지 먼저 확인
./tests/ctranslate2_test ../tests/data

# 2) Python wheel 빌드
cd ../python
CTRANSLATE2_ROOT=/usr/local python setup.py bdist_wheel
ls dist/                              # ctranslate2-X.Y.Z+pb1-...whl 확인

# 3) 분리된 venv-B에 설치
python -m venv /data/MyProject/stt/CTranslate2/tests-phrase-bias/.venv-B
source /data/MyProject/stt/CTranslate2/tests-phrase-bias/.venv-B/bin/activate
pip install -U pip
pip install dist/*.whl
pip install "faster-whisper" pyyaml jiwer

# 4) 기록: B가 phrase-bias 빌드(+pb suffix)인지 + API 존재 확인
python - <<'PY'
import ctranslate2
print("CT2(B)=", ctranslate2.__version__)   # +pb1 suffix 기대
# phrase_biases / phrase_bias_config API가 실제로 노출되는지 스모크 체크
from ctranslate2.models import Whisper
import inspect
# WhisperModel 래퍼 또는 generate 시그니처에 phrase_biases가 있는지 확인
print("phrase API present check below:")
PY
deactivate
```

> **CPU 전용 환경**이면 `-DWITH_CUDA=OFF -DWITH_CUDNN=OFF`로 빌드하고, A/B 모두 `device="cpu"`로 돌린다. A/B의 device·compute_type은 반드시 동일해야 한다.

> Version B wheel은 SSOT §D5대로 `X.Y.Z+pb1` 같은 suffix를 가진다. 이 suffix로 "지금 돌린 게 fork가 맞다"를 증명할 수 있어야 한다.

---

## 4. 테스트 데이터 준비

### 4.1 오디오 선정 기준
- **도메인 용어가 실제로 발화된** 파일만 넣는다. bias 대상 단어가 한 번도 안 나오면 recall 측정이 무의미하다.
- positive(정답 용어)·negative(대표 오인식) **양쪽을 자극**할 수 있게 구성한다. 예: baseline이 "트랜스포머"를 "트랜스퍼머"로 흘려듣는 클립을 일부러 포함.
- 길이·화자·녹음 환경을 다양화해 한 조건에 과적합되지 않게 한다.
- 최소 권장: 도메인 용어 출현 횟수 합계 **>= 50회** (recall 분모가 너무 작으면 통계가 흔들림).

### 4.2 정답 전사 형식 (`refs/<stem>.txt`)
- 오디오와 **같은 stem**, UTF-8, 1파일당 1전사(여러 줄 허용, 채점 시 합쳐 비교).
- 도메인 용어는 **정답 표기(surface)** 그대로 적는다. recall은 이 정답 표기를 기준으로 센다.
- 구두점·대소문자 정규화 정책을 정해 A/B/ref에 **동일하게** 적용한다(§6 `normalize()`).

```
# refs/sample_001.txt
오늘 발표에서 트랜스포머 구조의 어텐션을 설명했습니다.
```

### 4.3 도메인 vocab (`vocab/domain_terms.yaml`)

[`SSOT.md`](SSOT.md) §4 스키마를 따른다. 각 항목: `surface` / `aliases` / `policy(positive|negative)` / `mode(soft|block)` / `start_bias` / `step_bias` / `min_prefix_len`.

```yaml
# vocab/domain_terms.yaml  (SSOT §4 데이터 모델)
custom_vocabulary:
  # positive: 정답 도메인 용어 recall ↑  (start_bias > 0)
  - surface: "트랜스포머"
    aliases:
      - " 트랜스포머"      # Whisper tokenizer 앞 공백 민감성 대응
      - "트랜스 포머"
      - "Transformer"
      - " transformer"
    policy: positive
    mode: soft
    start_bias: 0.15
    step_bias: 0.60
    min_prefix_len: 1

  # negative: 대표 오인식 억제 (start_bias=0, step_bias<0 → 첫 토큰 과억제 방지)
  - surface: "트랜스퍼머"
    aliases:
      - " 트랜스퍼머"
    policy: negative
    mode: soft
    start_bias: 0.00
    step_bias: -0.80
    min_prefix_len: 2

runtime:
  per_domain_vocab: true
  enable_start_bias: true
  max_paths_per_surface: 16
```

> 가드레일([`SSOT.md`](SSOT.md) §5): `mode=block`에는 positive 값 금지, `abs(start_bias)/abs(step_bias)` 상한 준수, special token 금지, `min_prefix_len < len(ids)`. 채점기는 이 YAML의 `surface`(+`aliases`)를 "도메인 용어 사전"으로 그대로 읽어 recall/precision/misrecognition을 센다.

> negative 항목의 `surface`(예: "트랜스퍼머")는 **대표 오인식 목록**으로도 쓰인다. score.py가 misrecognition rate를 계산할 때 이 목록을 참조한다(§6).

---

## 5. 실행 절차

### 5.1 순서
1. venv-A 활성화 → `run_ab.py --version A` 로 전 오디오 전사 → `results/A/`
2. venv-B 활성화 → `run_ab.py --version B --vocab vocab/domain_terms.yaml` → `results/B/`
3. `score.py` 로 채점 → `results/scores.csv`, `results/summary.json`

```bash
BENCH=/data/MyProject/stt/CTranslate2/tests-phrase-bias

# --- Version A (bias OFF) ---
source $BENCH/.venv-A/bin/activate
python $BENCH/run_ab.py \
  --version A \
  --model  $BENCH/models/large-v3-ct2 \
  --audio  $BENCH/audio \
  --out    $BENCH/results/A
deactivate

# --- Version B (bias ON) ---
source $BENCH/.venv-B/bin/activate
python $BENCH/run_ab.py \
  --version B \
  --model  $BENCH/models/large-v3-ct2 \
  --audio  $BENCH/audio \
  --vocab  $BENCH/vocab/domain_terms.yaml \
  --out    $BENCH/results/B
deactivate

# --- 채점 ---
source $BENCH/.venv-B/bin/activate   # jiwer 있는 쪽 아무거나
python $BENCH/score.py \
  --refs   $BENCH/refs \
  --hyp_a  $BENCH/results/A \
  --hyp_b  $BENCH/results/B \
  --vocab  $BENCH/vocab/domain_terms.yaml \
  --out    $BENCH/results
deactivate
```

### 5.2 공정성 규칙 (반드시 준수)
- `beam_size`, `temperature`, `best_of`, `length_penalty`, `condition_on_previous_text`, `vad_filter`, `language` 등 **모든 디코딩 파라미터를 A/B 동일**하게 고정.
- `temperature=0`(또는 빈 fallback 튜플)으로 **deterministic** 디코딩. fallback 비활성 권장.
- A는 bias 인자를 **전달조차 하지 않는다**(미설치 API 호출 에러 방지 + 진짜 baseline).
- device / compute_type 동일.

### 5.3 `run_ab.py` 스켈레톤

```python
#!/usr/bin/env python
"""A/B 전사 러너. Version A=bias off, Version B=bias on (vocab 전달)."""
import argparse, glob, json, os, time
from faster_whisper import WhisperModel

# A/B 공통·고정 디코딩 파라미터 (공정성: 절대 분기시키지 말 것)
DECODE = dict(
    beam_size=5,
    temperature=0.0,            # deterministic
    best_of=1,
    length_penalty=1.0,
    condition_on_previous_text=False,
    vad_filter=False,
    language="ko",
)

def load_vocab_as_phrase_biases(vocab_path):
    """vocab/domain_terms.yaml -> fork API 인자.
    포크가 phrase_bias_config(YAML 경로)를 직접 받으면 그걸 쓰고,
    phrase_biases 리스트를 받으면 여기서 변환한다.
    실제 키 이름은 구현된 API에 맞춰 1곳만 고치면 됨.
    """
    return vocab_path  # phrase_bias_config 경로를 그대로 넘기는 경우

def build_model(version, model_dir, vocab_path):
    if version == "A":
        # baseline: bias 인자 전달 금지
        return WhisperModel(model_dir, device=DEVICE, compute_type=COMPUTE)
    # version B: 모델 init에 vocab baking (SSOT D8: 로드당 1회 compile)
    return WhisperModel(
        model_dir, device=DEVICE, compute_type=COMPUTE,
        phrase_bias_config=vocab_path,         # 또는 phrase_biases=[...]
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["A", "B"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--vocab", default=None)   # B에서만 사용
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="float16")
    args = ap.parse_args()

    global DEVICE, COMPUTE
    DEVICE, COMPUTE = args.device, args.compute_type
    os.makedirs(args.out, exist_ok=True)

    if args.version == "B" and not args.vocab:
        raise SystemExit("Version B requires --vocab")

    model = build_model(args.version, args.model, args.vocab)

    meta = {"version": args.version, "decode": DECODE,
            "device": DEVICE, "compute_type": COMPUTE,
            "vocab": args.vocab, "files": {}}

    for wav in sorted(glob.glob(os.path.join(args.audio, "*"))):
        stem = os.path.splitext(os.path.basename(wav))[0]
        t0 = time.time()
        segments, info = model.transcribe(wav, **DECODE)
        text = "".join(seg.text for seg in segments).strip()
        dt = time.time() - t0
        with open(os.path.join(args.out, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        meta["files"][stem] = {"sec": round(dt, 3), "chars": len(text)}

    with open(os.path.join(args.out, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
```

> **API 키 이름 주의**: 구현된 포크의 실제 키워드(`phrase_bias_config` vs `phrase_biases`)에 맞춰 `build_model()` / `load_vocab_as_phrase_biases()`의 한 줄만 고친다. 나머지 디코딩 경로는 A/B 공유.

---

## 6. 평가 지표 & 채점

[`SSOT.md`](SSOT.md) §5와 [`deep-research-report.md`](deep-research-report.md) §검증과 운영 로드맵의 지표를 구현한다.

### 6.1 측정 지표 정의

| 지표 | 정의 | 분모/분자 |
|------|------|-----------|
| **CER** | 문자 오류율 | edit_distance(chars) / ref_chars |
| **WER** | 단어 오류율 | edit_distance(words) / ref_words |
| **domain-term recall** | 정답에 있는 도메인 용어 중 hyp가 맞춘 비율 | Σ hit / Σ (ref 내 domain 용어 출현 수) |
| **domain-term precision** | hyp가 출력한 도메인 용어 중 정답에도 있던 비율 | Σ hit / Σ (hyp 내 domain 용어 출현 수) |
| **insertion ratio** | 삽입 오류 / ref 길이 | jiwer insertions / ref_words |
| **misrecognition rate** | 대표 오인식(negative surface)이 hyp에 남은 비율 | Σ (hyp에 남은 오인식) / Σ (해당 위치 발생 기대 수) |

> **핵심 해석**: phrase bias가 잘 동작하면 *recall ↑* 와 동시에 *misrecognition rate ↓* 이고, *insertion ratio* 는 거의 불변이어야 한다(§1 C1~C3).

### 6.2 domain-term hit 카운팅 방법
- vocab YAML의 각 항목에서 `policy: positive`인 `surface`(+그 표기 normalize)를 **정답 용어 집합**으로 둔다.
- `policy: negative`의 `surface`는 **대표 오인식 집합**으로 둔다.
- recall: ref 문자열에서 positive surface의 출현 횟수를 분모로, hyp에서도 같은 위치/횟수만큼 등장하면 hit.
  - MVP는 **count-based**(문자열 출현 횟수)로 단순화한다: `hit = min(count_in_ref, count_in_hyp)`.
- precision: hyp의 positive surface 총 출현 수를 분모로, ref에도 있으면 hit.
- misrecognition: ref에는 정답 용어가 있는데 hyp가 **negative surface**로 적은 경우를 센다.

### 6.3 `score.py` 스펙/스켈레톤

```python
#!/usr/bin/env python
"""results/A, results/B 채점. SSOT §5 지표."""
import argparse, csv, glob, json, os, re, unicodedata
import yaml
from jiwer import process_words, process_characters

def normalize(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[^\w가-힣 ]+", " ", s)      # 구두점 제거 (A/B/ref 동일 적용)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_terms(vocab_path):
    cfg = yaml.safe_load(open(vocab_path, encoding="utf-8"))
    pos, neg = [], []
    for item in cfg.get("custom_vocabulary", []):
        surf = normalize(item["surface"])
        (pos if item.get("policy") == "positive" else neg).append(surf)
    return pos, neg

def count(hay, needle):
    if not needle:
        return 0
    return len(re.findall(re.escape(needle), hay))

def read(d, stem):
    p = os.path.join(d, stem + ".txt")
    return normalize(open(p, encoding="utf-8").read()) if os.path.exists(p) else ""

def metrics_for(refs, hyp_dir, pos, neg):
    agg = dict(ref_w=0, ins=0, sub=0,
               rec_hit=0, rec_den=0, prec_hit=0, prec_den=0,
               misrec=0, misrec_den=0)
    wer_chunks, cer_chunks = [], []
    for ref_path in sorted(glob.glob(os.path.join(refs, "*.txt"))):
        stem = os.path.splitext(os.path.basename(ref_path))[0]
        ref = normalize(open(ref_path, encoding="utf-8").read())
        hyp = read(hyp_dir, stem)
        # WER/CER
        w = process_words(ref, hyp)
        wer_chunks.append((w.substitutions + w.deletions + w.insertions,
                           len(ref.split())))
        c = process_characters(ref, hyp)
        cer_chunks.append((c.substitutions + c.deletions + c.insertions,
                           len(ref.replace(" ", ""))))
        agg["ref_w"] += len(ref.split())
        agg["ins"]  += w.insertions
        agg["sub"]  += w.substitutions
        # domain recall / precision (count-based)
        for term in pos:
            r, h = count(ref, term), count(hyp, term)
            agg["rec_hit"]  += min(r, h); agg["rec_den"] += r
            agg["prec_hit"] += min(r, h); agg["prec_den"] += h
        # misrecognition: ref엔 정답용어, hyp엔 대표 오인식
        ref_has_domain = any(count(ref, t) for t in pos)
        for bad in neg:
            agg["misrec"] += count(hyp, bad)
            agg["misrec_den"] += count(ref, [t for t in pos][0] if pos else "") if False else 0
        if ref_has_domain:
            agg["misrec_den"] += sum(count(ref, t) for t in pos)
    def ratio(a, b): return round(a / b, 4) if b else 0.0
    tot_err = sum(e for e, _ in wer_chunks); tot_w = sum(n for _, n in wer_chunks)
    cer_err = sum(e for e, _ in cer_chunks); cer_c = sum(n for _, n in cer_chunks)
    return {
        "WER": ratio(tot_err, tot_w),
        "CER": ratio(cer_err, cer_c),
        "domain_recall":    ratio(agg["rec_hit"], agg["rec_den"]),
        "domain_precision": ratio(agg["prec_hit"], agg["prec_den"]),
        "insertion_ratio":  ratio(agg["ins"], agg["ref_w"]),
        "misrecognition_rate": ratio(agg["misrec"], agg["misrec_den"]),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--hyp_a", required=True)
    ap.add_argument("--hyp_b", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pos, neg = load_terms(a.vocab)
    mA = metrics_for(a.refs, a.hyp_a, pos, neg)
    mB = metrics_for(a.refs, a.hyp_b, pos, neg)

    # pass/fail (§1)
    crit = {
        "C1_recall_up":     mB["domain_recall"] - mA["domain_recall"] >= 0.05,
        "C2_misrec_down":   mB["misrecognition_rate"] <= mA["misrecognition_rate"],
        "C3_insertion_ok":  mB["insertion_ratio"] - mA["insertion_ratio"] <= 0.01,
        "C4_wer_ok":        mB["WER"] - mA["WER"] <= 0.005,
        "C5_precision_ok":  mB["domain_precision"] >= mA["domain_precision"] - 0.02,
    }
    summary = {"A": mA, "B": mB, "criteria": crit, "PASS": all(crit.values())}
    os.makedirs(a.out, exist_ok=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(a.out, "scores.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        keys = list(mA.keys())
        wcsv.writerow(["metric", "A", "B", "delta"])
        for k in keys:
            wcsv.writerow([k, mA[k], mB[k], round(mB[k]-mA[k], 4)])
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

> 위 misrecognition 카운팅은 MVP 근사다. 정밀 측정이 필요하면 위치 정렬(alignment) 기반으로 "정답 용어 자리에 오인식이 들어간 횟수"를 세도록 강화한다. 단순 시작점으로는 count-based로 충분하다.

### 6.4 예시 출력 표 (A vs B)

`results/scores.csv` 예시:

| metric | A (off) | B (on) | delta | 판정 |
|--------|--------:|-------:|------:|------|
| CER | 0.0721 | 0.0689 | -0.0032 | 양호 |
| WER | 0.1340 | 0.1318 | -0.0022 | C4 PASS |
| domain_recall | 0.620 | 0.840 | **+0.220** | C1 PASS |
| domain_precision | 0.910 | 0.905 | -0.005 | C5 PASS |
| insertion_ratio | 0.0180 | 0.0205 | +0.0025 | C3 PASS |
| misrecognition_rate | 0.310 | 0.090 | **-0.220** | C2 PASS |

`summary.json` → `"PASS": true` (C1~C5 모두 충족) → **효과 있음**.

---

## 7. 결과 해석

### "효과 있음" 패턴 (PASS)
- **domain_recall ↑ (C1)** + **misrecognition_rate ↓ (C2)**: positive/negative bias가 의도대로 동작.
- **insertion_ratio 거의 불변 (C3)** + **WER 비악화 (C4)** + **precision 유지 (C5)**: 품질 부작용 없음.
- 위 6.4 표가 전형적 합격 사례.

### 경고 신호 (over-biasing / FAIL)
| 증상 | 의미 | 조치 |
|------|------|------|
| recall ↑ **그러나** insertion_ratio 급증 (C3 실패) | bias가 없는 용어를 끼워 넣음(과편향) | positive `start_bias`/`step_bias` 낮추기, `min_prefix_len` ↑ |
| recall ↑ **그러나** precision ↓ (C5 실패) | 정답 용어를 엉뚱한 자리에 헛출력 | `step_bias` 위주로 조정, `start_bias`는 작게 |
| misrecognition ↓ **그러나** 정상 단어까지 사라짐 / WER ↑ (C4 실패) | negative bias 과억제 | `step_bias`(음수) 절댓값 낮추기, `min_prefix_len` ↑ |
| A와 B가 동일 출력 | bias 미적용(API 미연결, 모델 미일치, vocab 토큰화 실패) | §9 트러블슈팅 |

해석은 항상 §1 합격 기준으로 환원한다. **recall 단독 상승은 성공이 아니다** — [`SSOT.md`](SSOT.md) §5대로 insertion과 함께 봐야 PASS다.

---

## 8. 재현성 체크리스트

`env.lock.txt` 또는 `results/summary.json`에 아래를 기록한다.

- [ ] **모델 해시 고정**: `models/large-v3-ct2`의 `model.bin` SHA256 기록. A/B가 같은 경로/같은 해시인지 확인.
  ```bash
  sha256sum /data/MyProject/stt/CTranslate2/tests-phrase-bias/models/large-v3-ct2/model.bin
  ```
- [ ] **디코딩 파라미터 동일**: `run_ab.py`의 `DECODE` dict가 A/B 단일 출처. `_meta.json`에 덤프됨.
- [ ] **wheel 버전 기록**: A의 `ctranslate2.__version__`, B의 `...+pb1`. 둘 다 `env.lock.txt`에.
  ```bash
  pip freeze | grep -i ctranslate2     # 각 venv에서
  ```
- [ ] **deterministic seed**: `temperature=0`, fallback 비활성. (CT2 greedy/beam은 temp 0에서 결정적.)
- [ ] **하드웨어 기록**: GPU 모델(`nvidia-smi -L`), CUDA/cuDNN 버전, CPU, `compute_type(float16 등)`.
- [ ] **vocab 고정**: `domain_terms.yaml` SHA256 기록.
- [ ] **데이터셋 고정**: `audio/`, `refs/` 파일 목록·해시.
- [ ] **정규화 정책 동일**: `score.normalize()`를 ref/hyp_a/hyp_b에 동일 적용.

```bash
# env.lock.txt 생성 예
{
  echo "date=$(date -Iseconds)";
  echo "gpu=$(nvidia-smi -L 2>/dev/null | head -1)";
  echo "model_sha=$(sha256sum models/large-v3-ct2/model.bin | cut -d' ' -f1)";
  echo "vocab_sha=$(sha256sum vocab/domain_terms.yaml | cut -d' ' -f1)";
} > env.lock.txt
```

---

## 9. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| B 임포트 시 `undefined symbol` / CUDA 에러 | wheel이 잘못된 CUDA/cuDNN로 빌드됨 | 런타임 CUDA와 빌드 CUDA 일치 확인. `-DWITH_CUDA`/`-DWITH_CUDNN` 재확인 후 재빌드 |
| A == B (출력 완전 동일) | bias가 실제로 안 걸림 | (1) B의 `__version__`에 `+pb` suffix 있는지, (2) `phrase_bias_config`/`phrase_biases` 키 이름이 구현과 일치하는지, (3) vocab→token_paths 컴파일이 빈 결과인지(로그 확인) |
| recall 변화 없음인데 에러도 없음 | vocab surface가 모델 토크나이저로 토큰화 실패/빈 path | YAML surface가 실제 발화 표기와 맞는지, 앞 공백 alias 포함했는지 확인. token_paths가 비면 bias no-op |
| A/B의 baseline 전사가 서로 다름(같은 텍스트 부분도) | **모델/디코딩 파라미터 불일치** | 같은 `models/` 디렉토리·같은 `DECODE`·같은 `compute_type`인지 확인. 변환 모델이 양쪽 동일 해시인지 §8 |
| token_paths가 A/B에서 다르게 나옴 | 토크나이저 버전/파일 차이 | A/B venv의 `tokenizers`/`transformers` 버전 통일, 변환 시 `--copy_files tokenizer.json` 포함 확인 |
| insertion 폭증 | over-biasing | §7 경고 신호 표대로 bias 값 하향 |
| `jiwer`/`pyyaml` ImportError | 채점 의존성 누락 | 채점 돌리는 venv에 `pip install jiwer pyyaml` |
| 채점에서 한글이 깨짐 | 인코딩 | 모든 파일 UTF-8, `open(..., encoding="utf-8")` 확인 |
| B가 baseline보다 느림(latency ↑) | bias processor 오버헤드 | [`CLAUDE.md`](../CLAUDE.md) `--log_throughput`로 tokens/sec 비교. SSOT D7대로 disabled면 ≈0이어야 함 |

---

## 부록 — 한눈 실행 흐름

```
[모델 변환 1회] ──> models/large-v3-ct2 (A/B 공유)
        │
        ├─ venv-A (upstream)  ── run_ab.py --version A ──> results/A/
        ├─ venv-B (fork +pb1) ── run_ab.py --version B --vocab ──> results/B/
        │
        └─ score.py ──> scores.csv + summary.json(PASS/FAIL: C1~C5)
```

검증 설계의 근거와 지표 정의는 항상 [[SSOT.md]] §4(YAML/데이터 모델)·§5(가드레일/지표)로 돌아가 확인한다.
