#!/usr/bin/env bash
# sa_06 behavioural check — runs ON the target as root (via Target.run_check).
# The etc-backup.timer is installed + enabled, and triggering etc-backup.service
# actually produces a tar archive in /var/backups.
set -u

fail() { echo "FAIL: $*"; exit 1; }

systemctl list-unit-files 2>/dev/null | grep -q 'etc-backup.timer' \
  || fail "etc-backup.timer is not installed"
systemctl is-enabled --quiet etc-backup.timer || fail "etc-backup.timer is not enabled"
systemctl list-unit-files 2>/dev/null | grep -q 'etc-backup.service' \
  || fail "etc-backup.service is not installed"

before="$(ls /var/backups/*.tar* 2>/dev/null | wc -l)"
systemctl start etc-backup.service 2>/dev/null || fail "etc-backup.service failed to run"
# oneshot services return after completion; give a moment for the archive to land.
sleep 2
after="$(ls /var/backups/*.tar* 2>/dev/null | wc -l)"
[ "$after" -gt "$before" ] || fail "running etc-backup.service produced no tar archive in /var/backups"

echo "PASS: etc-backup.timer enabled and the service produces an archive in /var/backups"
