#!/bin/bash
# Snapshot this run directory to the private GitHub backup repo.
# Usage: bash backup.sh   (safe to run any time; no-op when nothing changed)
set -e
cd "$(dirname "$0")"
gh auth switch -u lehengzhangJacob >/dev/null 2>&1
trap 'gh auth switch -u cutcutjust >/dev/null 2>&1' EXIT
git add -A
if git diff --cached --quiet; then
  echo "nothing to commit"
  exit 0
fi
git commit --quiet -m "snapshot $(date -Is)" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
git push --quiet origin main
echo "pushed: $(git log --oneline -1)"
