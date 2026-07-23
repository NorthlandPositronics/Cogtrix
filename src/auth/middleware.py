"""RBAC enforcement middleware for FastAPI.

Provides ``require(permission)`` — a dependency factory that returns a
FastAPI ``Depends()`` callable.  Use it on any route to enforce that the
caller's role holds the required permission.

Example::

    from src.auth.middleware import require
    from src.auth.permissions import Permission

    @router.get("/items", dependencies=[Depends(require(Permission.ITEMS_READ))])
    async def list_items(): ...

Or as an injected dependency::

    async def list_items(
        current_user: TokenData = Depends(require(Permission.ITEMS_READ)),
    ): ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, HTTPException, status

from src.api.auth import TokenData, get_current_user
from src.auth.permissions import has_permission


def require(permission: str) -> Callable[..., Any]:
    """Return a FastAPI dependency that enforces *permission*.

    The dependency validates the bearer token (via ``get_current_user``) and
    then checks whether the user's role holds *permission*.  On success the
    dependency returns the ``TokenData`` so it can be used as an injected
    parameter.

    Args:
        permission: A permission string such as ``"sessions.create"``.

    Raises:
        HTTPException: 401 when no valid bearer token is present.
        HTTPException: 403 when the token is valid but the role lacks
            *permission*.  The detail includes the missing permission name.
    """

    async def _check(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if not has_permission(current_user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "FORBIDDEN",
                    "message": (
                        f"Authenticated user lacks permission '{permission}' for this action."
                    ),
                },
            )
        return current_user

    return _check
