#!/usr/bin/env bash
# sa_09 behavioural check — runs ON the target as root (via Target.run_check).
# Isolates systemd timer+service WIRING (the "sa_06-lite" scenario):
# heartbeat.timer is installed + enabled, heartbeat.service is installed, and
# triggering the service actually writes the marker file. NOT a backup-content
# check — only that the unit pair is wired correctly and the service runs.
# set -u (NOT pipefail): `cmd | grep -q` is SIGPIPE-unsafe under pipefail.
set -u

fail() { echo "FAIL: $*"; exit 1; }

systemctl list-unit-files 2>/dev/null | grep -q 'heartbeat.timer' \
  || fail "heartbeat.timer is not installed"
systemctl is-enabled --quiet heartbeat.timer || fail "heartbeat.timer is not enabled"
systemctl list-unit-files 2>/dev/null | grep -q 'heartbeat.service' \
  || fail "heartbeat.service is not installed"

# Prove the service really writes the marker: clear it, trigger the service,
# and confirm a fresh non-empty file appears (the dir must already exist from
# the agent's setup; a correct service creates it).
rm -f /var/lib/heartbeat/last-run 2>/dev/null
systemctl start heartbeat.service 2>/dev/null || fail "heartbeat.service failed to run"
# oneshot services return after completion; give a moment for the write to land.
sleep 2
[ -s /var/lib/heartbeat/last-run ] \
  || fail "running heartbeat.service did not write a non-empty /var/lib/heartbeat/last-run"

echo "PASS: heartbeat.timer enabled and the service writes /var/lib/heartbeat/last-run"
