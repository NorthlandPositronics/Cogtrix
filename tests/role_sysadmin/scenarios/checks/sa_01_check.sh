#!/usr/bin/env bash
# sa_01 behavioural check — runs ON the target as root (via Target.run_check).
# Asserts nginx is installed, active, enabled on boot, and actually serving the
# requested content on port 80. Exit 0 = task achieved; any failure exits non-zero
# with a FAIL line the scorecard surfaces.
set -u

fail() { echo "FAIL: $*"; exit 1; }

command -v nginx >/dev/null 2>&1 || fail "nginx is not installed"
systemctl is-active --quiet nginx || fail "nginx service is not active"
systemctl is-enabled --quiet nginx || fail "nginx is not enabled on boot"

body="$(curl -fsS http://localhost/ 2>/dev/null)" || fail "nginx is not serving on http://localhost/ (port 80)"
printf '%s' "$body" | grep -q "Cogtrix OK" || fail "served page body does not contain 'Cogtrix OK'"

echo "PASS: nginx active+enabled and serving 'Cogtrix OK' on port 80"
