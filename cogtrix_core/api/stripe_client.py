"""Lazy-loaded Stripe client factory (Enterprise Phase 1 — task 1.4.3).

Usage::

    stripe = get_stripe_client()
    session = stripe.checkout.Session.create(...)

The ``stripe`` package is imported on first call so that API servers that
do not need billing do not pay the import cost at startup.  The
``STRIPE_SECRET_KEY`` environment variable must be set before calling
``get_stripe_client()``; a ``RuntimeError`` is raised if it is absent.
"""

from __future__ import annotations

import os


def get_stripe_client():
    """Return a configured ``stripe`` module instance.

    Lazily imports the ``stripe`` package and sets ``stripe.api_key`` from
    the ``STRIPE_SECRET_KEY`` environment variable on every call.  This is
    safe because the Stripe library treats ``stripe.api_key`` as a
    module-level global, so subsequent calls simply re-affirm the key.

    Raises:
        RuntimeError: ``STRIPE_SECRET_KEY`` is not set in the environment.
        ImportError:  The ``stripe`` package is not installed
                      (install with ``pip install 'cogtrix[api]'``).
    """
    import stripe as _stripe  # type: ignore[import-untyped]

    key = os.environ.get("STRIPE_SECRET_KEY")
    if not key:
        raise RuntimeError("STRIPE_SECRET_KEY not configured")
    _stripe.api_key = key
    return _stripe
