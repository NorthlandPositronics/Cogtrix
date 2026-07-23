"""Cogtrix REST + WebSocket API package.

This package exposes the FastAPI application that powers the Cogtrix React
web frontend.  All business logic is implemented in the ``developer`` task;
this package contains only the contract skeleton — routes, schemas, auth
dependencies, and WebSocket dispatcher.

Version: v1
Base prefix: /api/v1
WebSocket prefix: /ws/v1
"""
