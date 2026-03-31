"""DSPy BaseLM subclass that routes completions through the claude CLI.

Uses ``claude -p`` (non-interactive print mode) so that users with a Claude
Code subscription can run plumb without a separate ANTHROPIC_API_KEY.

Pattern adapted from tinaudio/skills@b0cbd3d.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any

from dspy.clients.base_lm import BaseLM

from plumb import PlumbInferenceError


def find_claude_cli() -> str | None:
    """Return the path to the ``claude`` CLI binary, or None if not found."""
    return shutil.which("claude")


def _call_claude(prompt: str, model: str | None = None, timeout: int = 300) -> str:
    """Run ``claude -p`` with *prompt* on stdin and return the text response.

    Strips the ``CLAUDECODE`` env var to allow nesting inside a Claude Code
    session (the guard is for interactive terminal conflicts; programmatic
    subprocess usage is safe).
    """
    cmd = ["claude", "-p", "--output-format", "text"]
    if model:
        cmd.extend(["--model", model])

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        cwd=tempfile.gettempdir(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude -p exited {result.returncode}\nstderr: {result.stderr}"
        )
    return result.stdout


def _serialize_messages(
    prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
) -> str:
    """Convert a DSPy messages list into a single text prompt for the CLI.

    System messages get ``<system>`` tags, multi-turn conversations get
    ``[role]`` prefixes.  Single user messages are passed through as-is.
    """
    if not messages:
        return prompt or ""

    # Single user message — pass through without decoration
    if len(messages) == 1 and messages[0].get("role") == "user":
        return messages[0]["content"]

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<system>\n{content}\n</system>")
        else:
            parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _make_response(text: str, model: str) -> SimpleNamespace:
    """Build a minimal OpenAI-compatible response object for BaseLM."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, role="assistant"),
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        model=model,
    )


class ClaudeCodeLM(BaseLM):
    """DSPy LM that routes completions through the ``claude`` CLI."""

    def __init__(
        self,
        model: str = "sonnet",
        max_tokens: int = 28000,
        timeout: int = 300,
        **kwargs: Any,
    ):
        super().__init__(
            model=f"claude-code/{model}",
            model_type="chat",
            temperature=0.0,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.cli_model = model
        self.timeout = timeout

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        import sys

        text_input = _serialize_messages(prompt, messages)
        input_len = len(text_input)
        print(f"[ClaudeCodeLM] Calling claude -p --model {self.cli_model} ({input_len} chars)...", file=sys.stderr)
        response_text = _call_claude(text_input, model=self.cli_model, timeout=self.timeout)
        print(f"[ClaudeCodeLM] Got response ({len(response_text)} chars)", file=sys.stderr)
        return _make_response(response_text, self.model)
