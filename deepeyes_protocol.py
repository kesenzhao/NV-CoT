"""Protocol helpers shared by the DeepEyes benchmark evaluators.

The model is an agent: a complete tool or answer XML block ends one assistant
turn.  Treating the response as an unconstrained completion until the chat EOS
can expose arbitrary text after an otherwise valid terminal block and can feed
that text back into the next turn.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple


TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
TURN_STOP_SEQUENCES = [TOOL_CLOSE, ANSWER_CLOSE, "<|im_end|>"]


class ProtocolError(ValueError):
    """Raised when an assistant turn does not contain one valid terminal block."""


def _first_terminal_open(text: str) -> Tuple[Optional[str], int]:
    candidates = [
        ("tool", text.find(TOOL_OPEN)),
        ("answer", text.find(ANSWER_OPEN)),
    ]
    present = [(kind, index) for kind, index in candidates if index >= 0]
    return min(present, key=lambda item: item[1]) if present else (None, -1)


def normalize_protocol_response(
    text: str,
    *,
    finish_reason: Optional[str] = None,
) -> Tuple[str, str]:
    """Return exactly the first complete tool/answer block and its kind.

    OpenAI-compatible servers normally omit a matched stop string from the
    returned text.  If the completion stopped normally at a closing XML tag,
    restore that tag so the transcript remains a valid DeepEyes conversation.
    A length-truncated or otherwise unterminated response is rejected instead
    of being silently repaired.
    """

    if not isinstance(text, str) or not text.strip():
        raise ProtocolError("empty assistant response")

    kind, open_index = _first_terminal_open(text)
    if kind is None:
        raise ProtocolError("assistant response has no <tool_call> or <answer> block")

    close_tag = TOOL_CLOSE if kind == "tool" else ANSWER_CLOSE
    close_index = text.find(close_tag, open_index)
    if close_index >= 0:
        end_index = close_index + len(close_tag)
        return text[:end_index].rstrip(), kind

    if finish_reason == "stop":
        return text.rstrip() + close_tag, kind

    raise ProtocolError(
        f"unterminated {kind} block (finish_reason={finish_reason!r})"
    )


def parse_tool_call(text: str) -> dict[str, Any]:
    """Parse and validate the single normalized DeepEyes tool call."""

    start = text.find(TOOL_OPEN)
    end = text.find(TOOL_CLOSE, start + len(TOOL_OPEN))
    if start < 0 or end < 0:
        raise ProtocolError("missing complete tool-call block")
    try:
        action = json.loads(text[start + len(TOOL_OPEN) : end].strip())
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid tool-call JSON: {exc}") from exc
    if not isinstance(action, dict):
        raise ProtocolError("tool call must be a JSON object")
    if action.get("name") != "image_zoom_in_tool":
        raise ProtocolError(f"unsupported tool: {action.get('name')!r}")
    arguments = action.get("arguments")
    if not isinstance(arguments, dict):
        raise ProtocolError("tool-call arguments must be a JSON object")
    bbox = arguments.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ProtocolError("bbox_2d must contain exactly four coordinates")
    if any(not isinstance(value, (int, float)) for value in bbox):
        raise ProtocolError("bbox_2d coordinates must be numeric")
    return action


def extract_answer(text: str) -> str:
    """Extract a non-empty answer from a normalized assistant turn."""

    start = text.find(ANSWER_OPEN)
    end = text.find(ANSWER_CLOSE, start + len(ANSWER_OPEN))
    if start < 0 or end < 0:
        raise ProtocolError("missing complete answer block")
    answer = text[start + len(ANSWER_OPEN) : end].strip()
    if not answer:
        raise ProtocolError("empty final answer")
    return answer
