#!/usr/bin/env bash
# sa_05 behavioural check — runs ON the target as root (via Target.run_check).
# The heartbeat script exists + appends to its log when run, a schedule (cron or
# systemd timer) references it, and a valid logrotate config covers the log.
set -u

fail() { echo "FAIL: $*"; exit 1; }

[ -x /usr/local/bin/heartbeat.sh ] || fail "/usr/local/bin/heartbeat.sh missing or not executable"

count_lines() { [ -f "$1" ] && wc -l < "$1" 2>/dev/null || echo 0; }
before="$(count_lines /var/log/heartbeat.log)"
/usr/local/bin/heartbeat.sh >/dev/null 2>&1 || true
after="$(count_lines /var/log/heartbeat.log)"
[ "$after" -gt "$before" ] || fail "running heartbeat.sh did not append a line to /var/log/heartbeat.log"

scheduled=0
{
  crontab -l 2>/dev/null
  cat /etc/crontab 2>/dev/null
  cat /etc/cron.d/* 2>/dev/null
  for u in $(cut -d: -f1 /etc/passwd); do crontab -u "$u" -l 2>/dev/null; done
} | grep -q 'heartbeat' && scheduled=1
systemctl list-timers --all --no-legend 2>/dev/null | grep -qi 'heartbeat' && scheduled=1
systemctl list-unit-files 2>/dev/null | grep -qiE 'heartbeat.*\.timer' && scheduled=1
[ "$scheduled" -eq 1 ] || fail "no cron job or systemd timer schedules heartbeat"

conf="$(grep -rls '/var/log/heartbeat.log' /etc/logrotate.d/ /etc/logrotate.conf 2>/dev/null | head -1)"
[ -n "$conf" ] || fail "no logrotate config references /var/log/heartbeat.log"
logrotate -d "$conf" >/dev/null 2>&1 || fail "logrotate config '$conf' is invalid (logrotate -d failed)"

echo "PASS: heartbeat script works, is scheduled, and its log is covered by valid logrotate config"
