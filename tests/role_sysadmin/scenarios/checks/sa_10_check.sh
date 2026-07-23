#!/usr/bin/env bash
# sa_10 behavioural check — runs ON the target as root (via Target.run_check).
# Lightweight by design: this scenario exists to exercise compression under a
# large PROSE-reading workload for the context_max_tokens sweep (#2360/#2365), so
# task grading is secondary. We only confirm the agent produced a themes report
# that cites the incident post-mortems.
set -u

fail() { echo "FAIL: $*"; exit 1; }

REP=/root/themes.txt
[ -s "$REP" ] || fail "$REP missing or empty (agent did not write the themes report)"

hits=0
for n in 1 2 3 4 5 6 7 8; do
  grep -qi "incident_${n}" "$REP" && hits=$((hits + 1))
done
[ "$hits" -ge 3 ] || fail "report cites only $hits/8 incident files (expected >= 3)"

grep -qiE 'factor|theme|monitor|rollback|review|alert|redundan|capacity' "$REP" \
  || fail "report does not describe contributing factors"

echo "PASS: themes report present, cites $hits/8 incidents, describes contributing factors"
