#!/usr/bin/env bash
# sa_07 seed — runs ON the target as root BEFORE the agent (via Target._apply_seed).
# Installs nginx, then plants a config syntax error so the service fails to start.
# The agent must diagnose the bad config (root cause) and repair it.
set -e

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends nginx

# Make our broken server block the only thing defining :80 so the fault is
# unambiguous (remove the stock default site).
rm -f /etc/nginx/sites-enabled/default

# Missing semicolon after the `root` directive -> `nginx -t` fails -> won't start.
cat > /etc/nginx/conf.d/site.conf <<'EOF'
server {
    listen 80 default_server;
    server_name _;
    root /var/www/html
    location / {
        try_files $uri $uri/ =404;
    }
}
EOF

systemctl daemon-reload 2>/dev/null || true
# Leave the service in a failed/stopped state (restart attempt fails on bad config).
systemctl restart nginx 2>/dev/null || true
systemctl stop nginx 2>/dev/null || true

exit 0
