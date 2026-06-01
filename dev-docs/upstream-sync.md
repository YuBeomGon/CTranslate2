# Upstream Sync — 동기화 절차

이 포크를 OpenNMT/CTranslate2 본가와 동기화하는 방법과 주의점.

## 브랜치 전략

- `master` — upstream과 동기화하는 베이스 (직접 기능 개발 X)
- `feature/whisper-phrase-bias` — phrase bias 기능 개발 브랜치. 여기서 작업 후 개인 repo(`origin`) push

## upstream remote 등록 (최초 1회)

현재 `origin`은 본인 포크(`YuBeomGon/CTranslate2`)만 등록되어 있습니다.
본가를 동기화하려면 `upstream`을 추가하세요:

```bash
git remote add upstream https://github.com/OpenNMT/CTranslate2.git
git remote -v   # origin + upstream 확인
```

## 최신 변경 가져오기

```bash
git fetch upstream
git checkout master
git merge upstream/master      # 또는: git rebase upstream/master
git push origin master
```

## 충돌 최소화 원칙

- **내 문서는 `dev-docs/`와 `CLAUDE.md`에만** 둔다 → upstream에 없는 경로라 충돌이 안 난다.
- `docs/`(공식 Sphinx)는 수정하지 않는다.
- 코드 수정은 `dev-docs/changes.md`에 기록해두면, 충돌 해결 시 "내가 왜 이렇게 바꿨는지"를 빠르게 복기할 수 있다.

## 충돌이 났을 때

1. `git status`로 충돌 파일 확인
2. `dev-docs/changes.md`에서 해당 부분의 변경 의도 확인
3. 내 커스텀 의도를 유지하며 upstream 변경을 반영
4. 테스트 (`pytest tests/`, C++ 테스트) 후 커밋
