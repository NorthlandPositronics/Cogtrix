#!/usr/bin/env bash
# sa_07 behavioural check — runs ON the target as root (via Target.run_check).
# nginx config is valid again, the service is active, and port 80 responds with a
# real HTTP status (200/403/404 all mean nginx is serving; 000 = still down).
set -u

fail() { echo "FAIL: $*"; exit 1; }

nginx -t >/dev/null 2>&1 || fail "nginx config is still invalid (nginx -t fails)"
systemctl is-active --quiet nginx || fail "nginx service is not active"

code="$(curl -s -o /dev/null -w '%{http_code}' http://localhost/ 2>/dev/null)"
[ -n "$code" ] && [ "$code" != "000" ] || fail "nginx is not responding on port 80"

echo "PASS: nginx config valid, service active, responding on port 80 (HTTP $code)"
