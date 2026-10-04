"""Optional, best-effort notification after a committed terminal transition."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import structlog
from sqlalchemy import select

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import get_session_maker

logger = structlog.get_logger(__name__)
TerminalReason = Literal["execution_error", "execution_timeout", "lease_recovery_exhausted"]


@dataclass(frozen=True)
class TerminalRun:
    run_id: str
    thread_id: str
    user_id: str
    graph_id: str | None
    cron_id: str | None
    status: str
    occurred_at: datetime
    failure_code: TerminalReason | None


TerminalRunCallback = Callable[[TerminalRun], Awaitable[None]]
_callback: TerminalRunCallback | None = None


def set_terminal_run_callback(callback: TerminalRunCallback | None) -> None:
    """Register during application lifespan; None disables notification."""
    global _callback
    _callback = callback


async def notify_terminal_run(run_id: str, *, reason: TerminalReason | None = None) -> None:
    callback = _callback
    if callback is None:
        return
    # Callers invoke only after winning and committing a terminal transition.
    # Notification failures must never undo or replace that outcome.
    try:
        async with get_session_maker()() as session:
            run = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
            if run is None or run.status not in {"success", "error"}:
                return
            params = run.execution_params or {}
            metadata = params.get("run_metadata") or {}
            outcome = TerminalRun(
                run_id=run.run_id,
                thread_id=run.thread_id,
                user_id=run.user_id,
                graph_id=params.get("graph_id"),
                cron_id=metadata.get("aegra_cron_id"),
                status=run.status,
                occurred_at=run.updated_at,
                failure_code=(reason or "execution_error") if run.status == "error" else None,
            )
        await callback(outcome)
    except Exception as exc:
        logger.error("Terminal run callback failed", run_id=run_id, error_type=type(exc).__name__)
