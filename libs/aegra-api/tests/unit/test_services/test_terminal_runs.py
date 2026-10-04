from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aegra_api.services import terminal_runs, worker_executor
from aegra_api.services.run_preparation import _server_run_metadata


@pytest.fixture(autouse=True)
def reset_callback() -> None:
    terminal_runs.set_terminal_run_callback(None)


@pytest.mark.asyncio
async def test_reads_persisted_owner_and_server_cron_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    row = SimpleNamespace(
        run_id="run",
        thread_id="thread",
        user_id="123:456",
        status="error",
        updated_at=datetime.now(UTC),
        execution_params={"graph_id": "web", "run_metadata": {"aegra_cron_id": "cron"}},
    )
    session = AsyncMock()
    session.scalar.return_value = row
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(terminal_runs, "get_session_maker", lambda: MagicMock(return_value=context))
    callback = AsyncMock()
    terminal_runs.set_terminal_run_callback(callback)

    await terminal_runs.notify_terminal_run("run", reason="execution_timeout")

    outcome = callback.await_args.args[0]
    assert outcome.user_id == "123:456"
    assert outcome.cron_id == "cron"
    assert outcome.failure_code == "execution_timeout"
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_failure_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    row = SimpleNamespace(
        run_id="run",
        thread_id="thread",
        user_id="123:456",
        status="success",
        updated_at=datetime.now(UTC),
        execution_params={},
    )
    session = AsyncMock()
    session.scalar.return_value = row
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(terminal_runs, "get_session_maker", lambda: MagicMock(return_value=context))
    terminal_runs.set_terminal_run_callback(AsyncMock(side_effect=RuntimeError("transport failed")))

    await terminal_runs.notify_terminal_run("run")

    session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "running", "interrupted"])
async def test_nonfinal_outcomes_are_excluded(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    session = AsyncMock()
    session.scalar.return_value = SimpleNamespace(status=status)
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(terminal_runs, "get_session_maker", lambda: MagicMock(return_value=context))
    callback = AsyncMock()
    terminal_runs.set_terminal_run_callback(callback)

    await terminal_runs.notify_terminal_run("run")

    callback.assert_not_awaited()


def test_client_metadata_cannot_claim_an_automation() -> None:
    client_metadata = {"aegra_cron_id": "forged", "custom": "retained"}
    assert _server_run_metadata(client_metadata, None) == {"custom": "retained"}
    assert _server_run_metadata(client_metadata, "server-cron") == {
        "custom": "retained",
        "aegra_cron_id": "server-cron",
    }
    assert client_metadata["aegra_cron_id"] == "forged"


@pytest.mark.asyncio
@pytest.mark.parametrize("won", [True, False])
async def test_invalid_execution_state_notifies_only_the_winning_writer(
    monkeypatch: pytest.MonkeyPatch, won: bool
) -> None:
    session = AsyncMock()
    claimed = MagicMock(rowcount=1)
    failed = MagicMock()
    failed.scalar_one_or_none.return_value = "run" if won else None
    session.execute.side_effect = [claimed, failed]
    session.scalar.return_value = SimpleNamespace(execution_params=None)
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(worker_executor, "_get_session_maker", lambda: MagicMock(return_value=context))
    notify = AsyncMock()
    monkeypatch.setattr(worker_executor, "notify_terminal_run", notify)

    assert await worker_executor._acquire_and_load("run", "worker") is None

    if won:
        notify.assert_awaited_once_with("run", reason="execution_error")
    else:
        notify.assert_not_awaited()
