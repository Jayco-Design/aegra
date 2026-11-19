"""Test stream_mode='events' functionality.

Tests for issue #99: stream_mode="events" does not work
https://github.com/aegra/aegra/issues/99
"""

import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph

from aegra_api.services.graph_streaming import stream_graph_events


class _RecordingGraph:
    output_channels: list[str] | None = None

    def __init__(self) -> None:
        self.astream_kwargs: dict[str, Any] | None = None
        self.astream_events_kwargs: dict[str, Any] | None = None

    async def astream(self, _input_data: Any, _config: dict[str, Any], **kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
        self.astream_kwargs = kwargs
        yield "values", {"done": True}

    async def astream_events(
        self, _input_data: Any, _config: dict[str, Any], **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        self.astream_events_kwargs = kwargs
        yield {
            "event": "on_chain_stream",
            "run_id": "run-123",
            "data": {"chunk": ("values", {"done": True})},
        }


class TestEventsMode:
    """Test events mode streaming."""

    @pytest.mark.asyncio
    async def test_events_mode_yields_raw_events(self):
        """Test that stream_mode='events' yields raw events."""
        # Create a mock graph that yields astream_events
        mock_graph = MagicMock()

        # Mock astream_events to yield some events
        async def mock_astream_events(*args, **kwargs):
            # Yield a few different event types
            yield {
                "event": "on_chain_start",
                "name": "test_chain",
                "run_id": "run-123",
                "data": {"input": "test"},
            }
            yield {
                "event": "on_chain_stream",
                "name": "test_chain",
                "run_id": "run-123",
                "data": {"chunk": ("values", {"key": "value"})},
            }
            yield {
                "event": "on_chain_end",
                "name": "test_chain",
                "run_id": "run-123",
                "data": {"output": "result"},
            }

        # Return the async generator directly, not a coroutine
        mock_graph.astream_events = mock_astream_events

        # Mock get_context_jsonschema if it exists
        if hasattr(mock_graph, "get_context_jsonschema"):
            mock_graph.get_context_jsonschema = MagicMock(return_value={})

        config = {"run_id": "run-123", "metadata": {"run_attempt": 1}}
        input_data = {"messages": [{"role": "user", "content": "test"}]}

        # Stream with events mode
        events_yielded = []
        async for mode, payload in stream_graph_events(
            mock_graph,
            input_data,
            config,
            stream_mode=["events"],
        ):
            if mode == "events":
                events_yielded.append((mode, payload))

        # Should yield raw events
        assert len(events_yielded) > 0, "Expected at least one 'events' event to be yielded"

        # Check that we got the raw events
        event_types = [payload.get("event") for _, payload in events_yielded]
        assert "on_chain_start" in event_types or "on_chain_stream" in event_types or "on_chain_end" in event_types

    @pytest.mark.asyncio
    async def test_events_mode_with_on_chain_stream_events(self):
        """Test that on_chain_stream events are also yielded as raw events.

        This test reproduces issue #99: on_chain_stream events are processed
        but not yielded as raw events when stream_mode='events'.
        """
        mock_graph = MagicMock()

        async def mock_astream_events(*args, **kwargs):
            # Yield on_chain_stream events (these are the problematic ones)
            # These get processed but should ALSO be yielded as raw events
            yield {
                "event": "on_chain_stream",
                "name": "test_chain",
                "run_id": "run-123",
                "data": {"chunk": ("values", {"key": "value"})},
            }
            yield {
                "event": "on_chain_stream",
                "name": "test_chain",
                "run_id": "run-123",
                "data": {"chunk": ("debug", {"type": "checkpoint", "payload": {"tasks": []}})},
            }

        # Return the async generator directly, not a coroutine
        mock_graph.astream_events = mock_astream_events

        if hasattr(mock_graph, "get_context_jsonschema"):
            mock_graph.get_context_jsonschema = MagicMock(return_value={})

        config = {"run_id": "run-123", "metadata": {"run_attempt": 1}}
        input_data = {"messages": [{"role": "user", "content": "test"}]}

        # Stream with events mode ONLY (no other modes)
        events_yielded = []
        all_yielded = []
        async for mode, payload in stream_graph_events(
            mock_graph,
            input_data,
            config,
            stream_mode=["events"],  # Only events mode
        ):
            all_yielded.append((mode, payload))
            if mode == "events":
                events_yielded.append((mode, payload))

        # Debug: print what we got
        print(f"\nDEBUG: Total events yielded: {len(all_yielded)}")
        print(f"DEBUG: Event modes: {[m for m, _ in all_yielded]}")
        print(f"DEBUG: Raw 'events' yielded: {len(events_yielded)}")
        print(f"DEBUG: Events payloads event types: {[p.get('event') for _, p in events_yielded]}")

        # Check if we got on_chain_stream events as raw events
        on_chain_stream_events = [p for _, p in events_yielded if p.get("event") == "on_chain_stream"]

        # This test should FAIL with current implementation
        # on_chain_stream events are processed but not yielded as raw events
        # We should get at least 2 raw events (one for each on_chain_stream)
        assert len(on_chain_stream_events) >= 2, (
            f"BUG #99: Expected at least 2 raw 'events' with event='on_chain_stream', "
            f"but got {len(on_chain_stream_events)}. "
            f"Total raw events: {len(events_yielded)}, "
            f"Event types in raw events: {[p.get('event') for _, p in events_yielded]}. "
            f"on_chain_stream events are processed but not yielded as raw events when stream_mode='events'."
        )

    @pytest.mark.asyncio
    async def test_forwards_interrupts_to_astream(self) -> None:
        graph = _RecordingGraph()
        config = {
            "configurable": {"run_id": "run-123"},
            "metadata": {"run_attempt": 1},
            "interrupt_before": ["agent"],
            "interrupt_after": ["tools"],
        }

        async for _mode, _payload in stream_graph_events(
            graph,
            {"messages": []},
            config,
            stream_mode=["values"],
        ):
            pass

        assert graph.astream_kwargs is not None
        assert graph.astream_kwargs["interrupt_before"] == ["agent"]
        assert graph.astream_kwargs["interrupt_after"] == ["tools"]

    @pytest.mark.asyncio
    async def test_forwards_interrupts_to_astream_events(self) -> None:
        graph = _RecordingGraph()
        config = {
            "configurable": {"run_id": "run-123"},
            "metadata": {"run_attempt": 1},
            "interrupt_before": ["agent"],
            "interrupt_after": ["tools"],
        }

        async for _mode, _payload in stream_graph_events(
            graph,
            {"messages": []},
            config,
            stream_mode=["events"],
        ):
            pass

        assert graph.astream_events_kwargs is not None
        assert graph.astream_events_kwargs["interrupt_before"] == ["agent"]
        assert graph.astream_events_kwargs["interrupt_after"] == ["tools"]

    @pytest.mark.asyncio
    async def test_preserves_all_nodes_interrupt_sentinel(self) -> None:
        graph = _RecordingGraph()
        config = {
            "configurable": {"run_id": "run-123"},
            "metadata": {"run_attempt": 1},
            "interrupt_before": ["*"],
            "interrupt_after": ["*"],
        }

        async for _mode, _payload in stream_graph_events(
            graph,
            {"messages": []},
            config,
            stream_mode=["values"],
        ):
            pass

        assert graph.astream_kwargs is not None
        assert graph.astream_kwargs["interrupt_before"] == "*"
        assert graph.astream_kwargs["interrupt_after"] == "*"

    @pytest.mark.asyncio
    async def test_preserves_all_nodes_interrupt_sentinel_astream_events(self) -> None:
        graph = _RecordingGraph()
        config = {
            "configurable": {"run_id": "run-123"},
            "metadata": {"run_attempt": 1},
            "interrupt_before": ["*"],
            "interrupt_after": ["*"],
        }

        async for _mode, _payload in stream_graph_events(
            graph,
            {"messages": []},
            config,
            stream_mode=["events"],
        ):
            pass

        assert graph.astream_events_kwargs is not None
        assert graph.astream_events_kwargs["interrupt_before"] == "*"
        assert graph.astream_events_kwargs["interrupt_after"] == "*"


class _EventsGraph:
    """A graph whose astream_events yields the given raw events."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def astream_events(
        self, _input_data: Any, _config: dict[str, Any], **_kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        for event in self._events:
            yield event


async def _raw_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config = {"run_id": "run-123", "metadata": {"run_attempt": 1}}
    return [
        payload
        async for mode, payload in stream_graph_events(_EventsGraph(events), {}, config, stream_mode=["events"])
        if mode == "events"
    ]


class TestEventInputs:
    """Chain and chat-model events leave out their input, which repeats the whole state."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["on_chain_start", "on_chain_end", "on_chat_model_start", "on_chat_model_end"])
    async def test_chain_and_model_events_drop_their_input(self, kind: str) -> None:
        event = {
            "event": kind,
            "name": "node",
            "run_id": "child",
            "tags": ["t"],
            "metadata": {"m": 1},
            "data": {"input": {"messages": ["x" * 1000]}, "output": {"messages": ["reply"]}},
        }

        [payload] = await _raw_events([event])

        assert payload == {**event, "data": {"output": {"messages": ["reply"]}}}
        assert event["data"]["input"] == {"messages": ["x" * 1000]}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["on_tool_start", "on_tool_end", "on_chat_model_stream", "on_custom_event"])
    async def test_other_events_keep_their_input(self, kind: str) -> None:
        event = {"event": kind, "name": "tool", "run_id": "child", "data": {"input": {"key": 1}}}

        assert await _raw_events([event]) == [event]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("data", [None, "text", {}, {"output": "o"}])
    async def test_events_without_a_dict_input_pass_through(self, data: Any) -> None:
        event = {"event": "on_chain_end", "name": "node", "run_id": "child", "data": data}

        assert await _raw_events([event]) == [event]

    @pytest.mark.asyncio
    async def test_event_without_data_passes_through(self) -> None:
        event = {"event": "on_chain_start", "name": "node", "run_id": "child"}

        assert await _raw_events([event]) == [event]

    @pytest.mark.asyncio
    async def test_a_real_graph_streams_no_chain_or_model_input(self) -> None:
        model = GenericFakeChatModel(messages=iter([AIMessage("done")]))

        @tool
        def lookup(key: str) -> str:
            """Look a key up."""
            return f"value of {key}"

        async def agent(state: MessagesState) -> dict[str, Any]:
            await lookup.ainvoke({"key": "k"})
            return {"messages": [await model.ainvoke(state["messages"])]}

        graph = StateGraph(MessagesState).add_node("agent", agent).add_edge(START, "agent").compile()
        config = {"run_id": "run-123", "metadata": {"run_attempt": 1}}
        events = [
            payload
            async for mode, payload in stream_graph_events(
                graph, {"messages": [("user", "x" * 1000)]}, config, stream_mode=["events"]
            )
            if mode == "events"
        ]

        by_kind = {kind: [e for e in events if e["event"] == kind] for kind in {e["event"] for e in events}}
        for kind in ("on_chain_start", "on_chain_end", "on_chat_model_start", "on_chat_model_end"):
            assert by_kind.get(kind), f"the graph emitted no {kind}"
            assert all("input" not in (e.get("data") or {}) for e in by_kind[kind]), kind
        assert any("output" in e["data"] for e in by_kind["on_chat_model_end"])
        assert all(e["data"]["input"] == {"key": "k"} for e in by_kind["on_tool_start"])


def _root_chunk(mode: str) -> dict[str, Any]:
    return {"event": "on_chain_stream", "name": "graph", "run_id": "run-123", "data": {"chunk": (mode, {"m": mode})}}


async def _raw_root_modes(stream_mode: list[str]) -> list[str]:
    config = {"configurable": {"run_id": "run-123"}, "metadata": {"run_attempt": 1}}
    graph = _EventsGraph([_root_chunk(m) for m in ("values", "updates", "debug", "custom")])
    return [
        payload["data"]["chunk"][0]
        async for mode, payload in stream_graph_events(graph, {}, config, stream_mode=stream_mode)
        if mode == "events"
    ]


class TestRawStreamCopies:
    """The raw copy of a root stream chunk only goes out for a mode the client asked for."""

    @pytest.mark.asyncio
    async def test_debug_and_updates_copies_go_out_only_when_requested(self) -> None:
        assert await _raw_root_modes(["events", "values"]) == ["values", "custom"]
        assert await _raw_root_modes(["events", "values", "updates"]) == ["values", "updates", "custom"]
        assert await _raw_root_modes(["events", "debug", "updates"]) == ["values", "updates", "debug", "custom"]

    @pytest.mark.asyncio
    async def test_a_real_graph_streams_no_unrequested_root_copies(self) -> None:
        model = GenericFakeChatModel(messages=iter([AIMessage("done")]))

        async def agent(state: MessagesState) -> dict[str, Any]:
            return {"messages": [await model.ainvoke(state["messages"])]}

        graph = StateGraph(MessagesState).add_node("agent", agent).add_edge(START, "agent").compile()
        run_id = str(uuid.uuid4())
        config = {"run_id": run_id, "configurable": {"run_id": run_id}, "metadata": {"run_attempt": 1}}
        streamed = [
            (mode, payload)
            async for mode, payload in stream_graph_events(
                graph, {"messages": [("user", "hi")]}, config, stream_mode=["events", "values"]
            )
        ]

        root_copies = [
            p["data"]["chunk"][0]
            for mode, p in streamed
            if mode == "events" and p["event"] == "on_chain_stream" and p["run_id"] == run_id
        ]
        assert root_copies and set(root_copies) == {"values"}
        assert any(mode == "values" for mode, _ in streamed)
