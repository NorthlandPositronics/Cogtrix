# The "system" SUT for the Systems Administration role-test.
#
# A real systemd-managed Ubuntu box the agent configures over SSH. Run it with
#   docker run -d --rm --privileged --cgroupns=host -p <hostport>:22 <image>
# (systemd as PID 1 needs --privileged + a host cgroup namespace). The harness's
# tests/role_sysadmin/target.py:Target wraps build/run/key-injection/teardown.
#
# Validated locally (issue #2337): systemctl is-system-running == running,
# ssh-ready ~2s, and a full apt-install -> systemctl enable --now -> serve flow.
#
# The ``ops`` user has key-only SSH and passwordless sudo. authorized_keys is
# injected at RUNTIME (a fresh ephemeral keypair per run), so the image carries
# no embedded key and stays generic.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV container=docker

# Base sysadmin toolkit pre-installed so scenarios are fast + less network-flaky.
# apt lists are kept (not purged) so provisioning scenarios can install services
# (nginx, postgresql, ...) at runtime — the realistic "install it yourself" path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        systemd systemd-sysv dbus \
        openssh-server sudo ca-certificates \
        iproute2 iputils-ping curl ufw cron logrotate \
        python3 net-tools less vim-tiny \
 && useradd --create-home --shell /bin/bash ops \
 && usermod -aG sudo ops \
 && printf 'ops ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/90-ops \
 && chmod 0440 /etc/sudoers.d/90-ops \
 && install -d -o ops -g ops -m 0700 /home/ops/.ssh \
 && systemctl enable ssh \
 && systemctl mask systemd-networkd-wait-online.service 2>/dev/null || true

EXPOSE 22

# systemd's expected stop signal so `docker stop` shuts the box down cleanly.
STOPSIGNAL SIGRTMIN+3

CMD ["/sbin/init"]
