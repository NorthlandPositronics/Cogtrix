#!/usr/bin/env bash
# sa_08 behavioural check — runs ON the target as root (via Target.run_check).
# The widget.service is running again. (The fix is making its ExecStart script
# executable; the judge separately grades whether the agent named that root cause.)
set -u

fail() { echo "FAIL: $*"; exit 1; }

systemctl cat widget.service >/dev/null 2>&1 || fail "widget.service unit is missing"
systemctl is-active --quiet widget.service \
  || fail "widget.service is not active (status: $(systemctl is-active widget.service 2>/dev/null))"

echo "PASS: widget.service is active"
