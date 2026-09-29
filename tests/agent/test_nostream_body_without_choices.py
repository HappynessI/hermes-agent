"""A relay answering ``stream=True`` with a non-stream body carrying no choices must ride the
empty-stream retry path.

AgentRouter intermittently ignores ``stream=True`` and returns a plain body that has no
``choices`` at all. ``_adopt_final_response`` dereferenced ``final_response.choices``
unconditionally, so the turn died on ``'NoneType' object has no attribute 'choices'`` — an
opaque Python AttributeError that burned the retry budget and killed unattended runs (cron
deliveries, Bot Chat turns) instead of retrying the connection.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.errors import EmptyStreamError


def _agent_with(client):
    from run_agent import AIAgent

    with patch("agent.process_bootstrap.OpenAI", return_value=client):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://agentrouter.org/v1",
            provider="custom",
            model="deepseek-v4-flash",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def test_non_stream_body_without_choices_raises_empty_stream(monkeypatch):
    """The body that killed the run: no choices at all."""
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(id="body-without-choices", choices=None)

    with pytest.raises(EmptyStreamError):
        _agent_with(client)._interruptible_streaming_api_call({})


def test_non_stream_body_with_choices_still_switches_to_non_streaming():
    """A legitimate non-streaming answer keeps its behavior (and its content)."""
    client = MagicMock()
    message = SimpleNamespace(content="hello", tool_calls=None, reasoning_content=None, reasoning=None)
    client.chat.completions.create.return_value = SimpleNamespace(
        id="ok", choices=[SimpleNamespace(message=message, finish_reason="stop")])

    agent = _agent_with(client)
    response = agent._interruptible_streaming_api_call({})

    assert response.choices[0].message.content == "hello"
    assert agent._disable_streaming is True
