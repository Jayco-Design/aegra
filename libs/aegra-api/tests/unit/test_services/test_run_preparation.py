"""Unit tests for run_preparation helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from aegra_api.services import run_preparation as mod
from aegra_api.services.run_preparation import _validate_resume_command, update_thread_metadata


@pytest.fixture(autouse=True)
def _fast_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the resume-settle backoff so reject paths don't wait."""
    monkeypatch.setattr(mod, "_RESUME_SETTLE_INTERVAL_SECONDS", 0)


def _thread(status: str) -> SimpleNamespace:
    return SimpleNamespace(status=status)


def _session_returning(thread: object) -> AsyncMock:
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=thread)
    return session


def _patch_fresh_sessions(monkeypatch: pytest.MonkeyPatch, *threads: object) -> None:
    """Make run_preparation's fresh-session poll yield the given threads in order."""
    seq = list(threads)

    async def scalar(_stmt: object) -> object:
        return seq.pop(0) if len(seq) > 1 else (seq[0] if seq else None)

    fresh = AsyncMock()
    fresh.scalar = AsyncMock(side_effect=scalar)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=fresh)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "_get_session_maker", lambda: MagicMock(return_value=ctx))


class TestValidateResumeCommand:
    async def test_resume_on_interrupted_thread_passes(self) -> None:
        session = _session_returning(_thread("interrupted"))
        await _validate_resume_command(session, "t1", {"resume": "yes"})

    async def test_resume_none_on_non_interrupted_thread_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """{"resume": None} must still be guarded — None is a valid resume payload."""
        _patch_fresh_sessions(monkeypatch, _thread("idle"))
        session = _session_returning(_thread("idle"))
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "t1", {"resume": None})
        assert exc.value.status_code == 400

    async def test_resume_settles_when_status_flips_to_interrupted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The interrupt reaches the client before finalize commits 'interrupted';
        the guard polls a fresh session and accepts once the status settles."""
        _patch_fresh_sessions(monkeypatch, _thread("busy"), _thread("interrupted"))
        session = _session_returning(_thread("busy"))  # first (request-session) read is stale
        await _validate_resume_command(session, "t1", {"resume": "yes"})

    async def test_resume_on_missing_thread_is_404(self) -> None:
        session = _session_returning(None)
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "t1", {"resume": None})
        assert exc.value.status_code == 404

    async def test_non_resume_command_skips_check(self) -> None:
        session = _session_returning(_thread("idle"))
        await _validate_resume_command(session, "t1", {"goto": "node"})
        session.scalar.assert_not_awaited()

    async def test_none_command_skips_check(self) -> None:
        session = _session_returning(_thread("idle"))
        await _validate_resume_command(session, "t1", None)
        session.scalar.assert_not_awaited()


def _session_for_auto_create(*, raise_on_flush: bool = False) -> AsyncMock:
    """A session with no existing thread, ready to auto-create one.

    Mimics SQLAlchemy's begin_nested() savepoint contract precisely: flush()
    inside the `async with` block either succeeds or raises, and the context
    manager propagates that exception on exit (it never swallows it) — same
    as a real SAVEPOINT rollback.
    """
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=None)
    session.add = MagicMock()
    session.expunge = MagicMock()

    async def flush() -> None:
        if raise_on_flush:
            raise IntegrityError("insert", {}, Exception("duplicate key"))

    session.flush = AsyncMock(side_effect=flush)

    nested = MagicMock()
    nested.__aenter__ = AsyncMock(return_value=nested)
    nested.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested)
    return session


class TestUpdateThreadMetadataAutoCreateRace:
    """_prepare_run calls this for every run-start endpoint (create/stream/
    wait) to auto-create a thread that wasn't explicitly POSTed first — the
    same check-then-insert shape as api/threads.py's create_thread, and the
    same race: two concurrent run-starts for a brand-new thread_id can both
    pass the `not thread` check above and both attempt the insert."""

    async def test_creates_the_thread_when_none_exists(self) -> None:
        session = _session_for_auto_create()
        await update_thread_metadata(session, "t1", "a1", "g1", user_id="u1")
        session.add.assert_called_once()
        session.expunge.assert_not_called()

    async def test_losing_the_race_is_swallowed_not_raised(self) -> None:
        """The loser's insert conflicts with the winner's already-committed
        row (Postgres only raises a unique violation once the other
        transaction has concluded) — so it's safe to drop the loser's
        attempt and let the caller's later writes target the winner's row."""
        session = _session_for_auto_create(raise_on_flush=True)
        await update_thread_metadata(session, "t1", "a1", "g1", user_id="u1")
        session.expunge.assert_called_once()

    async def test_race_is_isolated_to_its_own_savepoint(self) -> None:
        """The failing insert must not touch the caller's outer transaction —
        confirmed by going through begin_nested(), not a bare flush/commit."""
        session = _session_for_auto_create(raise_on_flush=True)
        await update_thread_metadata(session, "t1", "a1", "g1", user_id="u1")
        session.begin_nested.assert_called_once()
        session.commit.assert_not_awaited()
