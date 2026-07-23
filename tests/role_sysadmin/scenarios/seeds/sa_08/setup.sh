#!/usr/bin/env bash
# sa_08 seed — runs ON the target as root BEFORE the agent (via Target._apply_seed).
# Installs a custom systemd service whose ExecStart script is NOT executable, so
# the unit fails to start with status=203/EXEC. The agent must find the permission
# root cause and fix it (e.g. chmod +x), not paper over the symptom.
set -e

install -d -m 0755 /opt/widget
cat > /opt/widget/run.sh <<'EOF'
#!/usr/bin/env bash
# A tiny long-running "service" process.
while true; do
    sleep 30
done
EOF

# The planted fault: the ExecStart target is not executable (or readable).
chmod 000 /opt/widget/run.sh

cat > /etc/systemd/system/widget.service <<'EOF'
[Unit]
Description=Widget service
After=network.target

[Service]
ExecStart=/opt/widget/run.sh
Restart=no

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable widget.service
# Start attempt fails with 203/EXEC because run.sh is not executable.
systemctl start widget.service 2>/dev/null || true

exit 0
