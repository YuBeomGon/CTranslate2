# AGENTS.md

이 저장소의 현재 작업 기준은 CTranslate2 Whisper KenLM BPE fusion 이식이다.

## SSOT

문서 우선순위는 다음과 같다.

1. `dev-docs/kenlm_fusion.md`
   - 현재 설계, 정책, IO 계약, build/API, benchmark, release readiness의 SSOT
2. `dev-docs/README.md`
   - dev-docs 문서 지도
3. `dev-docs/archive/`
   - 구현 계획, 체크리스트, phase별 리뷰, 과거 reference 보관. authoritative 문서가 아님

## Links

- KenLM fusion SSOT: `dev-docs/kenlm_fusion.md`
- Documentation map: `dev-docs/README.md`
- Archive: `dev-docs/archive/`

## Documentation Rules

- 최신 사용자/릴리즈 정책은 `dev-docs/kenlm_fusion.md`에만 정의한다.
- 구현 계획, 체크리스트, phase별 리뷰, reference는 `dev-docs/archive/`에 보존한다.
- archive 문서는 근거로만 사용한다. 최신 판단과 충돌하면 `dev-docs/kenlm_fusion.md`가 우선한다.
- 코드 동작이나 지원 범위가 바뀌면 관련 문서를 같은 작업에서 갱신한다.

## Coding Notes

- 기본 동작은 feature-off baseline parity를 유지해야 한다.
- hardcoded threshold, path, model name, timeout, feature flag는 피하고 옵션으로 분리한다.
- Python 파일을 추가하거나 수정할 때는 프로젝트 규칙에 맞는 모듈 헤더를 둔다.
- 테스트를 생략하면 작업 설명에 생략 사유를 남긴다.
