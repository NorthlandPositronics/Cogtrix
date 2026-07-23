"""Systems Administration holistic role-test (local-only).

The agent configures a disposable systemd container (the SUT) over SSH using
Cogtrix's real ``execute_shell_command``; the harness grades against the live
system state. See README.md and issue #2337.
"""
