"""CLI entry point for the Cogtrix API server.

Usage:
    python -m src.api                          # defaults
    python -m src.api --debug                  # debug logging to stdout/stderr
    python -m src.api --log                    # info logging to cogtrix-api.log
    python -m src.api --log-file /tmp/api.log  # custom log file
    python -m src.api --config-file prod.yaml  # explicit config
    python -m src.api --host 0.0.0.0 --port 9000

The docker entrypoint delegates here when invoked with ``api``.
"""

from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser (without calling parse_args)."""
    parser = argparse.ArgumentParser(
        prog="cogtrix-api",
        description="Cogtrix REST + WebSocket API server",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("COGTRIX_API_HOST", "0.0.0.0"),  # nosec B104
        help="Bind host (default: $COGTRIX_API_HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("COGTRIX_API_PORT", "8000")),
        help="Bind port (default: $COGTRIX_API_PORT or 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("COGTRIX_API_WORKERS", "1")),
        help="Number of uvicorn workers (default: 1)",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default=os.environ.get("COGTRIX_CONFIG_FILE"),
        help="Path to config file (JSON or YAML)",
    )
    parser.add_argument(
        "--log",
        nargs="?",
        const="",
        default=None,
        help="Enable logging (default file: cogtrix-api.log)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log to a specific file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=bool(os.environ.get("COGTRIX_DEBUG")),
        help="Debug mode (implies --log, verbose output)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (development)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Resolve log destination:
    #   --log-file <path>          → file (highest priority)
    #   --log [<path>]             → file (default: cogtrix-api.log)
    #   --debug (no --log-file)    → stdout (DEBUG/INFO) + stderr (WARNING+)
    #   (nothing)                  → logging disabled
    log_file: str | None = None
    stream_output: bool = False
    if args.log_file is not None:
        log_file = args.log_file
    elif args.log is not None:
        log_file = args.log or "cogtrix-api.log"
    elif args.debug:
        stream_output = True  # --debug without explicit file → split streams

    # Propagate config file via env var so app.py lifespan picks it up
    if args.config_file:
        os.environ["COGTRIX_CONFIG_FILE"] = args.config_file

    # Propagate logging settings via env vars for the lifespan
    if log_file is not None:
        os.environ["COGTRIX_API_LOG_FILE"] = log_file
    if args.debug:
        os.environ["COGTRIX_DEBUG"] = "1"
    if stream_output:
        os.environ["COGTRIX_LOG_STREAM"] = "1"

    # Set up Cogtrix logging before uvicorn starts
    try:
        from cogtrix_core.logging_config import setup_logging

        setup_logging(
            log_file=log_file,
            debug=args.debug,
            console_output=True,
            verbose=args.debug,
            stream_output=stream_output,
        )
    except Exception as exc:
        print(f"Warning: could not initialize logging: {exc}", file=sys.stderr)

    import uvicorn

    uvicorn.run(
        "cogtrix_core.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()
