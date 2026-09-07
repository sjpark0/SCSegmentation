#!/bin/sh
# 스테이징된 추가 줄에 토큰처럼 보이는 문자열이 있으면 커밋을 막습니다.
# 설치: cp docs/tools/pre-commit-secrets.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
# 배경: docs/operations.md 함정 8 (2026-09-07, GitHub 푸시 보호에 걸린 사례)
pat='hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|gho_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|ntn_[A-Za-z0-9]{30,}|secret_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY'
hits=$(git diff --cached -U0 --diff-filter=AM --no-color \
       | grep -E '^\+' | grep -v '^+++ ' \
       | grep -E -o "$pat" | sed -E 's/^(.{11}).*/\1…/' | sort -u)
if [ -n "$hits" ]; then
  echo "pre-commit: 비밀값처럼 보이는 문자열이 스테이징돼 있어 커밋을 막습니다 (docs/operations.md 함정 8):" >&2
  echo "$hits" | sed 's/^/  /' >&2
  echo "  값을 가린 뒤 다시 커밋하십시오. 의도된 것이면: git commit --no-verify" >&2
  exit 1
fi
exit 0
