# AGENTS.md

이 저장소의 현재 작업 기준은 CTranslate2 Whisper KenLM BPE fusion 이식이다.

## SSOT

문서 우선순위는 다음과 같다.

1. `dev-docs/ct2_kenlm_fusion_design.md`
   - 설계, 정책, IO 계약, 지원 범위의 SSOT
2. `dev-docs/ct2_kenlm_fusion_implementation_plan.md`
   - 실제 CTranslate2 코드에 적용하는 실행 계획
3. `dev-docs/ct2_kenlm_fusion_checklist.md`
   - 진행 상태 체크리스트
4. `dev-docs/reference/`
   - 리서치와 과거 초안 보관. authoritative 문서가 아님

## Links

- Design SSOT: `dev-docs/ct2_kenlm_fusion_design.md`
- Implementation plan: `dev-docs/ct2_kenlm_fusion_implementation_plan.md`
- Checklist: `dev-docs/ct2_kenlm_fusion_checklist.md`
- References: `dev-docs/reference/`

## Documentation Rules

- 설계 결정은 design 문서에만 정의한다.
- implementation plan은 design 결정을 반복 정의하지 않고 코드 적용 위치와 순서를 설명한다.
- checklist는 상태만 관리하고 새로운 정책을 정의하지 않는다.
- reference 문서는 근거로만 사용한다. 최신 판단과 충돌하면 design 문서가 우선한다.
- 코드 동작이나 지원 범위가 바뀌면 관련 문서를 같은 작업에서 갱신한다.

## Coding Notes

- 기본 동작은 feature-off baseline parity를 유지해야 한다.
- hardcoded threshold, path, model name, timeout, feature flag는 피하고 옵션으로 분리한다.
- Python 파일을 추가하거나 수정할 때는 프로젝트 규칙에 맞는 모듈 헤더를 둔다.
- 테스트를 생략하면 작업 설명에 생략 사유를 남긴다.
