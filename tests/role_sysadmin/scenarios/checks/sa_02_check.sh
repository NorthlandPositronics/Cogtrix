#!/usr/bin/env bash
# sa_02 behavioural check — runs ON the target as root (via Target.run_check).
# PostgreSQL installed + accepting connections + enabled on boot, and the
# appuser/appdb role+database exist and the role can actually connect over TCP.
set -u

fail() { echo "FAIL: $*"; exit 1; }

command -v psql >/dev/null 2>&1 || fail "postgresql is not installed (psql missing)"
pg_isready -h 127.0.0.1 -q || fail "postgres is not accepting TCP connections on 127.0.0.1"
systemctl is-enabled --quiet postgresql || fail "postgresql is not enabled on boot"

sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='appuser'" 2>/dev/null \
  | grep -q 1 || fail "login role 'appuser' does not exist"
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='appdb'" 2>/dev/null \
  | grep -q 1 || fail "database 'appdb' does not exist"

PGPASSWORD='Cogtrix-pw-2026' psql -h 127.0.0.1 -U appuser -d appdb -tAc 'SELECT 1' 2>/dev/null \
  | grep -q 1 || fail "appuser cannot connect to appdb over TCP with the expected password"

echo "PASS: postgresql up+enabled; appuser connects to appdb over 127.0.0.1"
