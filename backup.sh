#!/bin/bash
# Snapshot this run directory to the private GitHub backup repo.
# Usage: bash backup.sh ["commit message"]   (safe any time; no-op when nothing changed)
# Pushes with an explicit token for lehengzhangJacob -- no global account switching,
# so concurrent calls from the research runs and the sidecar cannot race.
set -e
cd "$(dirname "$0")"
MSG="${1:-snapshot $(date -Is)}"
git add -A
if git diff --cached --quiet; then
  echo "nothing to commit"
  exit 0
fi
git commit --quiet -m "$MSG" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
TOKEN=$(gh auth token -u lehengzhangJacob)
git -c credential.helper= -c http.extraheader="Authorization: Basic $(printf 'x-access-token:%s' "$TOKEN" | base64 -w0)" push --quiet origin main
echo "pushed: $(git log --oneline -1)"
