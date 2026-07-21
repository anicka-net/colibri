#!/usr/bin/env python3
"""Dependency-free OpenAI-compatible HTTP gateway for the colibri engine."""

import argparse
import codecs
import collections
import contextlib
import datetime
import json
import math
import mimetypes
import os
import select
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


HERE = Path(__file__).resolve().parent
END = b"\x01\x01END\x01\x01\n"
READY = b"\x01\x01READY\x01\x01\n"
MAX_BODY = 4 << 20
PROFILE_TURNS = 120           # rolling window of per-turn PROF snapshots kept for /profile
RESPONSE_HISTORY_ENTRIES = 32
RESPONSE_HISTORY_BYTES = 16 << 20
PROMPT_BYTES_PER_TOKEN_LIMIT = 16
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://tauri.localhost",
    "tauri://localhost",
)


class APIError(Exception):
    def __init__(self, status, message, param=None, code=None, error_type="invalid_request_error",
                 headers=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type
        self.headers = headers or {}


class ClientCancelled(Exception):
    pass


def error_object(error):
    return {"error": {"message": error.message, "type": error.error_type,
                      "param": error.param, "code": error.code}}


def anthropic_error_object(error, request_id):
    error_type = {"server_error": "api_error"}.get(error.error_type, error.error_type)
    return {"type": "error", "error": {"type": error_type,
                                           "message": error.message},
            "request_id": request_id}


def ollama_error_object(error):
    return {"error": error.message}


class GenerationScheduler:
    """Bounded FIFO admission for the engine's independent KV contexts."""

    def __init__(self, max_queue=8, queue_timeout=300, capacity=1):
        if max_queue < 0:
            raise ValueError("max_queue cannot be negative")
        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.capacity = capacity
        self.free_slots = set(range(capacity))
        self.condition = threading.Condition()
        self.queue = collections.deque()
        self.active = 0
        self.closed = False
        self.admitted = 0
        self.completed = 0
        self.rejected = 0
        self.timed_out = 0
        self.cancelled = 0

    @contextlib.contextmanager
    def admit(self, cancelled=None, slot=None):
        ticket = object()
        queued_at = time.monotonic()
        with self.condition:
            if self.closed:
                raise APIError(503, "The inference scheduler is shutting down.", None,
                               "scheduler_closed", "server_error")
            if (self.active >= self.capacity or self.queue) and len(self.queue) >= self.max_queue:
                self.rejected += 1
                raise APIError(429, "The inference queue is full.", None, "queue_full",
                               "rate_limit_error", {"Retry-After": "1"})
            self.queue.append(ticket)
            deadline = queued_at + self.queue_timeout
            while True:
                if self.closed:
                    self.queue.remove(ticket)
                    self.condition.notify_all()
                    raise APIError(503, "The inference scheduler is shutting down.", None,
                                   "scheduler_closed", "server_error")
                available = min(self.free_slots) if slot is None and self.free_slots else slot
                if self.queue[0] is ticket and available in self.free_slots:
                    break
                if cancelled and cancelled():
                    self.queue.remove(ticket)
                    self.cancelled += 1
                    self.condition.notify_all()
                    raise ClientCancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.queue.remove(ticket)
                    self.timed_out += 1
                    self.condition.notify_all()
                    raise APIError(429, "Timed out waiting for the inference engine.", None,
                                   "queue_timeout", "rate_limit_error", {"Retry-After": "1"})
                self.condition.wait(min(remaining, 0.25))
            self.queue.popleft()
            self.free_slots.remove(available)
            self.active += 1
            self.admitted += 1
            wait_seconds = time.monotonic() - queued_at
        try:
            yield wait_seconds, available
        finally:
            with self.condition:
                self.active -= 1
                self.free_slots.add(available)
                self.completed += 1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"active": self.active, "queued": len(self.queue),
                    "capacity": self.capacity,
                    "max_queue": self.max_queue, "queue_timeout_seconds": self.queue_timeout,
                    "admitted": self.admitted, "completed": self.completed,
                    "rejected": self.rejected, "timed_out": self.timed_out,
                    "cancelled": self.cancelled}

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def content_text(content, param):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise APIError(400, "Message content must be a string or an array of text parts.", param)
    parts = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
            raise APIError(400, "Colibri currently supports text message content only.",
                           f"{param}.{index}", "unsupported_content_type")
        if not isinstance(part.get("text"), str):
            raise APIError(400, "Text content parts require a string `text` field.",
                           f"{param}.{index}.text")
        parts.append(part["text"])
    return "".join(parts)


def responses_messages(body, previous=()):
    messages = list(previous)
    instructions = body.get("instructions")
    if instructions is not None:
        messages.append({"role": "system", "content": content_text(instructions, "instructions")})
    value = body.get("input")
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise APIError(400, "Responses `input` items must be message objects.",
                               f"input.{index}")
            item_type = item.get("type")
            if item_type == "function_call":
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    raise APIError(400, "Responses function calls require a string `name`.",
                                   f"input.{index}.name")
                messages.append({"role": "assistant", "content": None, "tool_calls": [{
                    "id": item.get("call_id") or item.get("id") or "call_compat",
                    "type": "function",
                    "function": {"name": name,
                                 "arguments": item.get("arguments", "{}")},
                }]})
            elif item_type == "function_call_output":
                output = item.get("output", "")
                messages.append({"role": "tool",
                                 "content": output if isinstance(output, str)
                                 else json.dumps(output, ensure_ascii=False),
                                 "tool_call_id": item.get("call_id")})
            elif item.get("role") in ("system", "developer", "user", "assistant"):
                content = item.get("content")
                if isinstance(content, list):
                    parts = []
                    for part_index, part in enumerate(content):
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            parts.append(part.get("text", ""))
                        else:
                            parts.append(content_text([part],
                                f"input.{index}.content.{part_index}"))
                    content = "".join(parts)
                else:
                    content = content_text(content, f"input.{index}.content")
                messages.append({"role": item["role"], "content": content})
            else:
                raise APIError(400, "Unsupported Responses input item.",
                               f"input.{index}", "unsupported_input_type")
    else:
        raise APIError(400, "`input` must be a string or an array of messages.", "input")
    return messages


def anthropic_messages(body):
    messages = []
    system = body.get("system")
    if system is not None:
        messages.append({"role": "system", "content": content_text(system, "system")})
    value = body.get("messages")
    if not isinstance(value, list) or not value:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    for index, item in enumerate(value):
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            raise APIError(400, "Anthropic messages require user or assistant roles.",
                           f"messages.{index}.role")
        content = item.get("content")
        if not isinstance(content, list):
            messages.append({"role": item["role"],
                             "content": content_text(content, f"messages.{index}.content")})
            continue
        text, calls, results = [], [], []
        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                raise APIError(400, "Message content blocks must be objects.",
                               f"messages.{index}.content.{part_index}")
            part_type = part.get("type")
            if part_type == "text":
                text.append(content_text([part], f"messages.{index}.content"))
            elif part_type in ("thinking", "redacted_thinking"):
                continue
            elif part_type == "tool_use" and item["role"] == "assistant":
                calls.append({"id": part.get("id") or "call_compat", "type": "function",
                              "function": {"name": part.get("name"),
                                           "arguments": json.dumps(part.get("input", {}),
                                                                   ensure_ascii=False)}})
            elif part_type == "tool_result" and item["role"] == "user":
                result = content_text(part.get("content", ""), f"messages.{index}.content")
                results.append({"role": "tool", "content": result,
                                "tool_call_id": part.get("tool_use_id")})
            else:
                raise APIError(400, "Unsupported Anthropic content block.",
                               f"messages.{index}.content.{part_index}",
                               "unsupported_content_type")
        if item["role"] == "assistant":
            message = {"role": "assistant", "content": "".join(text) or None}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
        else:
            text_index = result_index = 0
            for part in content:
                if part.get("type") == "text":
                    messages.append({"role": "user", "content": text[text_index]})
                    text_index += 1
                elif part.get("type") == "tool_result":
                    messages.append(results[result_index])
                    result_index += 1
    return messages


def anthropic_tools(tools):
    result = []
    for index, tool in enumerate(tools or []):
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise APIError(400, "Anthropic tools require a string `name`.", f"tools.{index}")
        result.append({"type": "function", "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        }})
    return result


def ollama_messages(messages):
    """Translate Ollama chat messages to the internal OpenAI-shaped representation."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    result = []
    for index, item in enumerate(messages):
        if not isinstance(item, dict) or item.get("role") not in (
                "system", "developer", "user", "assistant", "tool"):
            raise APIError(400, "Unsupported Ollama message role.", f"messages.{index}.role")
        if item.get("images"):
            raise APIError(400, "GLM-5.2 Colibri is text-only; Ollama images are unsupported.",
                           f"messages.{index}.images", "unsupported_content_type")
        message = {"role": item["role"],
                   "content": content_text(item.get("content", ""),
                                           f"messages.{index}.content")}
        calls = item.get("tool_calls")
        if calls:
            normalized = []
            for call_index, call in enumerate(calls):
                fn = call.get("function") if isinstance(call, dict) else None
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    raise APIError(400, "Ollama tool calls require a function name.",
                                   f"messages.{index}.tool_calls.{call_index}")
                arguments = fn.get("arguments", {})
                normalized.append({"id": call.get("id") or f"call_ollama_{call_index}",
                                   "type": "function", "function": {
                                       "name": fn["name"],
                                       "arguments": (arguments if isinstance(arguments, str)
                                                     else json.dumps(arguments, ensure_ascii=False))}})
            message["tool_calls"] = normalized
        result.append(message)
    return result


def ollama_options(body, server_max_tokens):
    options = body.get("options") or {}
    if not isinstance(options, dict):
        raise APIError(400, "`options` must be an object.", "options")
    normalized = dict(body)
    if "num_predict" in options:
        normalized["max_tokens"] = options["num_predict"]
    if "temperature" in options:
        normalized["temperature"] = options["temperature"]
    if "top_p" in options:
        normalized["top_p"] = options["top_p"]
    maximum, temperature, top_p = generation_options(normalized, server_max_tokens)
    return normalized, maximum, temperature, top_p


# ---- GLM-5.2 tool calling -----------------------------------------------------------------
# The model expresses tool calls as ordinary text (from chat_template.jinja):
#   <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
# and tool results come back as <|observation|><tool_response>{content}</tool_response>.
# We render those markers into the prompt and parse them back into OpenAI `tool_calls`.
import re

BOX_START, BOX_END = "<tool_call>", "</tool_call>"
TR_OPEN,  TR_CLOSE = "<tool_response>", "</tool_response>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
DS4_PUBLIC_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")
DS4_HIDDEN_MODELS = ("deepseek-chat",)

_BOX_RE  = re.compile(re.escape(BOX_START) + r"(.*?)" + re.escape(BOX_END), re.DOTALL)
_ARG_RE  = re.compile(r"<arg_key>([^<]*)</arg_key><arg_value>(.*?)</arg_value>", re.DOTALL)
_NAME_RE = re.compile(r"\s*([A-Za-z0-9_.\-]+)")
_TAG_RE  = re.compile(r"</?arg_key>|</?arg_value>")

# De-mangler: opt-in recovery for heavily-quantized models that drop the
# <arg_key>K</arg_key><arg_value> structure. Default OFF (never rewrites well-formed output).
_SALVAGE = os.environ.get("COLI_TOOL_SALVAGE", "0") == "1"


def _tool_param_order(tools):
    """name -> ordered param names (required first) from the request schema, for de-mangling."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        params = ((fn.get("parameters") or {}).get("properties") or {})
        required = list((fn.get("parameters") or {}).get("required") or [])
        out[name] = required + [p for p in params if p not in required]
    return out


def _tool_param_types(tools):
    """name -> {param: declared JSON-schema type}. The model emits every argument as text;
    without the schema a string-typed value that happens to look numeric ("12345" for an
    order id, an SKU, a phone number) would be json.loads()'d into an int and the tool would
    receive the wrong type."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        types = {}
        for key, spec in props.items():
            if isinstance(spec, dict):
                t = spec.get("type")
                if isinstance(t, list):          # {"type": ["string", "null"]}
                    t = next((x for x in t if x != "null"), None)
                types[key] = t
        out[name] = types
    return out


def _coerce_arg(value, declared):
    """Decode a raw <arg_value> according to the declared schema type.

    A string-typed parameter is kept verbatim -- never parsed as JSON. Everything else keeps
    the previous permissive behaviour (parse if it parses, otherwise leave as text)."""
    if declared == "string":
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    if declared in ("integer", "number") and isinstance(parsed, bool):
        return value                              # `true` is not a number
    if declared and declared not in ("integer", "number", "boolean", "object", "array"):
        return value
    return parsed


def parse_tool_calls(reply, tools=None):
    """Return (content, tool_calls). Strict GLM parse; optional de-mangler (COLI_TOOL_SALVAGE=1)
    rescues malformed int4 output by mapping a lone payload onto the tool's primary parameter."""
    param_order = _tool_param_order(tools)
    param_types = _tool_param_types(tools)
    calls, salvaged = [], []
    for match in _BOX_RE.finditer(reply):
        inner = match.group(1)
        name_match = _NAME_RE.match(inner)
        name = name_match.group(1) if name_match else inner.strip()
        args = {}
        types = param_types.get(name, {})
        for arg in _ARG_RE.finditer(inner):
            key, value = arg.group(1), arg.group(2)
            args[key] = _coerce_arg(value, types.get(key))
        if not args and _SALVAGE:
            rest = inner[name_match.end():] if name_match else ""
            payload = _TAG_RE.sub("", rest).strip()
            if payload.startswith("(") and payload.endswith(")"):
                payload = payload[1:-1].strip()
            if payload:
                key = (param_order.get(name) or ["input"])[0]
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                args = {key: payload}
                salvaged.append(name)
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    text = _BOX_RE.sub("", reply)
    if THINK_CLOSE in text:
        text = text.split(THINK_CLOSE, 1)[1]
    text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    if calls:
        dm = len(salvaged)
        sys.stderr.write("[api] tool-calls: %d total, %d strict, %d de-mangled [%s]%s\n"
                         % (len(calls), len(calls) - dm, dm, "CLEAN" if dm == 0 else "DE-MANGLED",
                            (" -> " + ", ".join(salvaged)) if dm else ""))
        sys.stderr.flush()
    return text.strip(), calls


def reasoning_settings(body, default=False):
    """Normalize the thinking controls used by OpenAI, DeepSeek, and Anthropic clients."""
    enabled = default and body.get("model") not in DS4_HIDDEN_MODELS
    effort = body.get("reasoning_effort")
    if effort is None and isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort")
    efforts = (None, "none", "minimal", "low", "medium", "high", "xhigh", "max")
    if effort not in efforts:
        raise APIError(400, "`reasoning_effort` must be none, minimal, low, medium, high, xhigh, or max.",
                       "reasoning_effort")
    if effort is not None:
        enabled = effort != "none"

    for field in ("enable_thinking", "think"):
        if field in body:
            if not isinstance(body[field], bool):
                raise APIError(400, f"`{field}` must be a boolean.", field)
            enabled = body[field]

    if "thinking" in body:
        thinking = body["thinking"]
        if not isinstance(thinking, dict) or thinking.get("type") not in ("enabled", "disabled"):
            raise APIError(400, "`thinking.type` must be enabled or disabled.", "thinking")
        enabled = thinking["type"] == "enabled"
    return enabled, effort or "high"


def split_reasoning(reply, enabled):
    """Split a complete GLM reply into hidden reasoning and user-visible content."""
    if not enabled:
        return "", reply.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    if THINK_CLOSE not in reply:
        return reply.replace(THINK_OPEN, "").strip(), ""
    reasoning, content = reply.split(THINK_CLOSE, 1)
    return (reasoning.replace(THINK_OPEN, "").replace(THINK_CLOSE, "").strip(),
            content.replace(THINK_OPEN, "").replace(THINK_CLOSE, "").strip())


class ReasoningStream:
    """Route a streamed reply around a possibly chunk-split </think> marker."""

    def __init__(self, enabled, on_reasoning, on_content):
        self.reasoning = enabled
        self.on_reasoning = on_reasoning
        self.on_content = on_content
        self.pending = ""
        self.content_pending = ""

    def feed_content(self, text):
        self.content_pending += text
        cleaned = self.content_pending.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
        hold = max(len(THINK_OPEN), len(THINK_CLOSE)) - 1
        if len(cleaned) > hold:
            self.on_content(cleaned[:-hold])
            self.content_pending = cleaned[-hold:]
        else:
            self.content_pending = cleaned

    def feed(self, text):
        if not self.reasoning:
            if self.content_pending or THINK_OPEN in text or THINK_CLOSE in text:
                self.feed_content(text)
            else:
                self.on_content(text)
            return
        self.pending += text
        marker = self.pending.find(THINK_CLOSE)
        if marker >= 0:
            before = self.pending[:marker].replace(THINK_OPEN, "")
            if before:
                self.on_reasoning(before)
            after = self.pending[marker + len(THINK_CLOSE):]
            self.pending = ""
            self.reasoning = False
            if after:
                self.feed_content(after)
            return
        hold = len(THINK_CLOSE) - 1
        if len(self.pending) > hold:
            emit, self.pending = self.pending[:-hold], self.pending[-hold:]
            emit = emit.replace(THINK_OPEN, "")
            if emit:
                self.on_reasoning(emit)

    def close(self):
        if self.pending:
            (self.on_reasoning if self.reasoning else self.on_content)(
                self.pending.replace(THINK_OPEN, "").replace(THINK_CLOSE, ""))
            self.pending = ""
        if self.content_pending:
            self.on_content(self.content_pending.replace(THINK_OPEN, "").replace(THINK_CLOSE, ""))
            self.content_pending = ""


def render_chat(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                tool_choice=None):
    """Render the text-only subset of the official GLM-5.2 chat template."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    prompt = ["[gMASK]<sop>"]
    if enable_thinking:
        effort = "Max" if reasoning_effort in ("xhigh", "max") else "High"
        prompt.append(f"<|system|>Reasoning Effort: {effort}")
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or tool_choice.get("name"))
        if forced:
            tools = [t for t in (tools or [])
                     if ((t.get("function", t) if isinstance(t, dict) else {}).get("name") == forced)]
    elif tool_choice == "none":
        tools = None                              # the client forbade tools: do not offer them
    if tools:
        # AUTHORITATIVE GLM-5.2 tool-declaration block (byte-matches chat_template.jinja): the
        # `# Tools` + <tools></tools> XML structure is what the model was trained on. A made-up
        # preamble makes it hallucinate other frameworks' syntax (e.g. `end_action`).
        prompt.append("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the "
                      "user query.\n\nYou are provided with function signatures within <tools></tools> "
                      "XML tags:\n<tools>\n")
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            clean = {k: v for k, v in fn.items() if k not in ("defer_loading", "strict")}
            prompt.append(json.dumps(clean, ensure_ascii=False) + "\n")
        prompt.append("</tools>\n\nFor each function call, output the function name and arguments "
                      "within the following XML format:\n<tool_call>{function-name}"
                      "<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
                      "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")
        if forced:
            prompt.append(f"\n\nYou must call the function `{forced}`. Do not answer directly.")
        elif tool_choice == "required":
            prompt.append("\n\nYou must call one of the functions above. Do not answer directly.")
    prev_tool = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role in ("system", "developer"):
            prompt.append(f"<|system|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "user":
            prompt.append(f"<|user|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "assistant":
            # content may be null when the message is purely tool_calls
            raw = message.get("content")
            text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            prompt.append(f"<|assistant|><think></think>{text.strip()}")
            for tc in (message.get("tool_calls") or []):
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                prompt.append(BOX_START + (fn.get("name") or ""))
                for key, value in (args or {}).items():
                    prompt.append(f"<arg_key>{key}</arg_key><arg_value>"
                                  + (value if isinstance(value, str)
                                     else json.dumps(value, ensure_ascii=False)) + "</arg_value>")
                prompt.append(BOX_END)
        elif role == "tool":
            if not prev_tool:                       # one <|observation|> per consecutive tool run
                prompt.append("<|observation|>")
            prompt.append(TR_OPEN + content_text(message.get("content"), f"messages.{index}.content") + TR_CLOSE)
        else:
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
        prev_tool = (role == "tool")
    prompt.append("<|assistant|><think>" if enable_thinking else
                  "<|assistant|><think></think>")
    return "".join(prompt)


def generation_options(body, limit):
    if body.get("n", 1) != 1:
        raise APIError(400, "Colibri currently supports `n=1` only.", "n", "unsupported_value")
    # `tools`/`functions` are handled by render_chat (declaration) + parse_tool_calls (output).
    choice = body.get("tool_choice")
    if choice is not None:
        if isinstance(choice, str):
            if choice not in ("auto", "none", "required"):
                raise APIError(400, "`tool_choice` must be one of \"auto\", \"none\", \"required\", "
                                    "or a function object.", "tool_choice", "unsupported_value")
        elif isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name") or choice.get("name")
            if not name:
                raise APIError(400, "`tool_choice` function object must include a name.",
                               "tool_choice", "invalid_value")
            declared = [(t.get("function", t) if isinstance(t, dict) else {}).get("name")
                        for t in (body.get("tools") or body.get("functions") or [])]
            if name not in declared:
                raise APIError(400, f"`tool_choice` names {name!r}, which is not in `tools`.",
                               "tool_choice", "invalid_value")
        else:
            raise APIError(400, "`tool_choice` must be a string or a function object.",
                           "tool_choice", "invalid_value")
        if choice != "none" and not (body.get("tools") or body.get("functions")):
            raise APIError(400, "`tool_choice` requires `tools`.", "tool_choice", "invalid_value")
    if body.get("stop") is not None:
        raise APIError(400, "Custom stop sequences are not supported yet.", "stop", "unsupported_parameter")
    if body.get("logprobs"):
        raise APIError(400, "Log probabilities are not supported yet.", "logprobs", "unsupported_parameter")
    if body.get("frequency_penalty", 0) or body.get("presence_penalty", 0):
        raise APIError(400, "Token penalties are not supported yet.", None, "unsupported_parameter")
    if body.get("seed") is not None:
        raise APIError(400, "Per-request seeds are not supported yet.", "seed", "unsupported_parameter")
    response_format = body.get("response_format")
    if response_format not in (None, {"type": "text"}):
        raise APIError(400, "Only the default text response format is supported.",
                       "response_format", "unsupported_parameter")

    maximum = body.get("max_completion_tokens")
    maximum_param = "max_completion_tokens"
    if maximum is None:
        maximum = body.get("max_tokens")
        maximum_param = "max_tokens"
    if maximum is None:
        maximum = min(256, limit)
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    temperature = 0.7 if temperature is None else temperature
    top_p = 0.9 if top_p is None else top_p
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise APIError(400, f"`{maximum_param}` must be a positive integer.", maximum_param)
    if maximum > limit:
        maximum = limit   # clamp to the server's --max-tokens cap instead of 400 (#260): OpenAI
                          # clients (opencode/ai-sdk) default to large max_tokens; rejecting breaks them.
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
            not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise APIError(400, "`temperature` must be between 0 and 2.", "temperature")
    if (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or
            not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise APIError(400, "`top_p` must be greater than 0 and at most 1.", "top_p")
    return maximum, float(temperature), float(top_p)


def read_engine_turn(stream, sentinel, on_bytes):
    pending = b""
    while True:
        byte = stream.read(1)
        if byte == b"":
            raise RuntimeError("colibri engine exited unexpectedly")
        pending += byte
        if pending.endswith(sentinel):
            data = pending[:-len(sentinel)]
            if data:
                on_bytes(data)
            break
        if len(pending) > len(sentinel):
            on_bytes(pending[:-len(sentinel)])
            pending = pending[-len(sentinel):]

    fields = stream.readline().decode("utf-8", "replace").strip().split()
    if len(fields) < 5 or fields[0] != "STAT":
        raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
    return {
        "completion_tokens": int(fields[1]),
        "tokens_per_second": float(fields[2]),
        "cache_hit_percent": float(fields[3]),
        "rss_gb": float(fields[4]),
        "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
        "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
    }


class Engine:
    def __init__(self, executable, model, cap=8, max_tokens=1024, env=None, kv_slots=1,
                 expert_bits=8, dense_bits=None):
        child_env = dict(env or os.environ, SNAP=str(model), SERVE="1", SERVE_BATCH="1",
                         NGEN=str(max_tokens), KV_SLOTS=str(kv_slots))
        dense_bits = expert_bits if dense_bits is None else dense_bits
        self.process = subprocess.Popen(
            [str(executable), str(cap), str(expert_bits), str(dense_bits)],
            env=child_env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, bufsize=0,
        )
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.next_request_id = 1
        self.closed = False
        self.dispatcher_error = None
        self.kv_slots = kv_slots
        self.tiers = None
        self.hwinfo = None
        self.emap = None
        self.hits = None
        self.hits_seq = 0                      # latest "TIERS" snapshot from the engine
        self.profile = collections.deque(maxlen=PROFILE_TURNS)  # per-turn phase timings
        self.profile_seq = 0
        read_engine_turn(self.process.stdout, READY, lambda _: None)
        self.dispatcher = threading.Thread(target=self._dispatch_stdout,
                                           name="colibri-stdout", daemon=True)
        self.dispatcher.start()

    @staticmethod
    def _stats(fields):
        if len(fields) < 5 or fields[0] != "STAT":
            raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
        return {
            "completion_tokens": int(fields[1]),
            "tokens_per_second": float(fields[2]),
            "cache_hit_percent": float(fields[3]),
            "rss_gb": float(fields[4]),
            "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
            "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
        }

    def _fail_pending(self, error):
        with self.pending_lock:
            requests = list(self.pending.values())
            self.pending.clear()
        for events in requests:
            events.put(("error", error))

    def _read_exact(self, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if chunk == b"":
                raise RuntimeError("truncated engine DATA payload")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _dispatch_stdout(self):
        try:
            while True:
                line = self.process.stdout.readline()
                if line == b"":
                    raise RuntimeError("colibri engine exited unexpectedly")
                fields = line.decode("utf-8", "replace").strip().split()
                if not fields:
                    continue
                kind = fields[0]
                if kind == "DATA" and len(fields) == 3:
                    request_id = fields[1]
                    size = int(fields[2])
                    if not 0 <= size <= 65536:
                        raise RuntimeError("invalid engine DATA size")
                    data = self._read_exact(size)
                    if self._read_exact(1) != b"\n":
                        raise RuntimeError("invalid engine DATA terminator")
                    with self.pending_lock:
                        events = self.pending.get(request_id)
                    if events is not None:
                        events.put(("data", data))
                elif kind == "DONE" and len(fields) >= 7:
                    request_id = fields[1]
                    stats = self._stats(fields[2:])
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                    if events is not None:
                        events.put(("done", stats))
                elif kind == "HWINFO" and len(fields) >= 7:
                    parts = " ".join(fields[6:]).split("|")
                    self.hwinfo = {"cores": int(fields[1]), "ram_total_gb": float(fields[2]),
                                   "ram_avail_gb": float(fields[3]), "gpus": int(fields[4]),
                                   "vram_total_gb": float(fields[5]),
                                   "cpu": parts[0].strip() if len(parts)>0 else "",
                                   "gpu": parts[1].strip() if len(parts)>1 else ""}
                elif kind == "EMAP" and len(fields) == 4:
                    self.emap = {"rows": int(fields[1]), "cols": int(fields[2]), "map": fields[3]}
                elif kind == "HITS" and len(fields) == 4:
                    self.hits = fields[3]
                    self.hits_seq += 1
                elif kind == "PROF" and len(fields) >= 10:
                    # per-turn phase timings: where the engine spent this turn's wall time
                    self.profile.append({
                        "wall_s": float(fields[1]),
                        "prompt_tokens": int(fields[2]),
                        "completion_tokens": int(fields[3]),
                        "expert_disk_s": float(fields[4]),
                        "expert_wait_s": float(fields[5]),
                        "expert_matmul_s": float(fields[6]),
                        "attention_s": float(fields[7]),
                        "lm_head_s": float(fields[8]),
                        "forwards": int(fields[9]),
                    })
                    self.profile_seq += 1
                elif kind == "TIERS" and len(fields) >= 6:
                    self.tiers = {"vram": int(fields[1]), "ram": int(fields[2]),
                                  "disk": int(fields[3]), "vram_gb": float(fields[4]),
                                  "ram_gb": float(fields[5])}
                elif kind == "ERROR" and len(fields) >= 2:
                    request_id = fields[1]
                    message = " ".join(fields[2:]) or "engine request failed"
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                    if events is not None:
                        events.put(("error", RuntimeError(message)))
                else:
                    raise RuntimeError(f"invalid engine response: {' '.join(fields)}")
        except Exception as error:
            if not self.closed:
                self.dispatcher_error = error
                self._fail_pending(error)

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None):
        if isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or not 0 <= cache_slot < self.kv_slots:
            raise APIError(400, "Invalid cache slot.", "cache_slot")
        payload = prompt.encode("utf-8")
        if b"\0" in payload:
            raise APIError(400, "NUL bytes are not supported in prompts.", "messages")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def decode(data):
            text = decoder.decode(data)
            if text:
                on_text(text)

        events = queue.Queue()
        with self.pending_lock:
            if self.closed:
                raise RuntimeError("colibri engine is shutting down")
            if self.dispatcher_error is not None:
                raise RuntimeError("colibri engine dispatcher stopped") from self.dispatcher_error
            if self.process.poll() is not None:
                raise RuntimeError("colibri engine is not running")
            request_id = str(self.next_request_id)
            self.next_request_id += 1
            self.pending[request_id] = events
        header = (f"SUBMIT {request_id} {cache_slot} {len(payload)} {max_tokens} "
                  f"{temperature:.8g} {top_p:.8g}\n").encode()
        try:
            with self.write_lock:
                if self.process.poll() is not None:
                    raise RuntimeError("colibri engine is not running")
                self.process.stdin.write(header + payload + b"\n")
                self.process.stdin.flush()
        except Exception:
            with self.pending_lock:
                self.pending.pop(request_id, None)
            raise

        cancel_sent = False
        while True:
            try:
                kind, value = events.get(timeout=0.25)
            except queue.Empty:
                if not cancel_sent and cancelled and cancelled():
                    cancel_sent = True
                    with self.write_lock:
                        self.process.stdin.write(f"CANCEL {request_id}\n".encode())
                        self.process.stdin.flush()
                continue
            if kind == "data":
                if not cancel_sent:
                    decode(value)
                    if cancelled and cancelled():
                        cancel_sent = True
                        with self.write_lock:
                            self.process.stdin.write(f"CANCEL {request_id}\n".encode())
                            self.process.stdin.flush()
            elif kind == "done":
                tail = decoder.decode(b"", final=True)
                if tail:
                    on_text(tail)
                return value
            elif cancel_sent and isinstance(value, RuntimeError) and str(value) == "CANCELLED":
                raise ClientCancelled()
            else:
                raise value

    def close(self):
        with self.pending_lock:
            if self.closed:
                return
            self.closed = True
        self._fail_pending(RuntimeError("colibri engine is shutting down"))
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.dispatcher is not threading.current_thread():
            self.dispatcher.join(timeout=5)


def model_object(model_id, created, context_length=None, max_tokens=None):
    created_at = datetime.datetime.fromtimestamp(created, datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z")
    # Deliberately a superset accepted by OpenAI/OpenRouter and Anthropic clients.
    model = {"id": model_id, "object": "model", "type": "model", "created": created,
             "created_at": created_at, "display_name": "GLM-5.2 (Colibri)",
             "owned_by": "colibri"}
    if context_length:
        model.update({
            "name": "GLM-5.2 (Colibri)",
            "context_length": context_length,
            "top_provider": {"context_length": context_length,
                             "max_completion_tokens": max_tokens or context_length,
                             "is_moderated": False},
            "supported_parameters": ["tools", "tool_choice", "max_tokens", "temperature",
                                     "top_p", "stream", "reasoning_effort"],
        })
    return model


class APIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine, model_id, api_key=None, max_tokens=1024,
                 cors_origins=DEFAULT_CORS_ORIGINS, max_queue=8, queue_timeout=300,
                 kv_slots=1, model_aliases=(), hidden_model_aliases=(),
                 context_length=None, default_thinking=False, model_path=None):
        super().__init__(address, APIHandler)
        self.engine = engine
        self.model_id = model_id
        self.model_ids = tuple(dict.fromkeys((model_id, *model_aliases)))
        self.hidden_model_ids = tuple(dict.fromkeys(hidden_model_aliases))
        self.accepted_model_ids = frozenset((*self.model_ids, *self.hidden_model_ids))
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.context_length = context_length
        self.default_thinking = default_thinking
        self.scheduler = GenerationScheduler(max_queue, queue_timeout, kv_slots)
        self.kv_slots = kv_slots
        self.cors_origins = tuple(cors_origins)
        self.created = int(time.time())
        self.model_size = 0
        self.model_modified = 0
        if model_path:
            if os.path.isfile(model_path):
                try:
                    model_stat = os.stat(model_path)
                except OSError:
                    pass
                else:
                    self.model_size = model_stat.st_size
                    self.model_modified = int(model_stat.st_mtime)
            else:
                for root, dirs, files in os.walk(model_path):
                    dirs[:] = [name for name in dirs if not name.startswith(".coli")]
                    for name in files:
                        if name.startswith(".coli"):
                            continue
                        try:
                            file_stat = os.stat(os.path.join(root, name))
                        except OSError:
                            continue
                        self.model_size += file_stat.st_size
                        self.model_modified = max(
                            self.model_modified, int(file_stat.st_mtime))
        self.watchdog_lock = threading.Lock()
        self.watchdog_active = 0
        self.response_history_lock = threading.Lock()
        self.response_history = collections.OrderedDict()
        self.response_history_bytes = 0

    @contextlib.contextmanager
    def watchdog_request(self, active):
        if active:
            with self.watchdog_lock:
                self.watchdog_active += 1
        try:
            yield
        finally:
            if active:
                with self.watchdog_lock:
                    self.watchdog_active -= 1

    def watchdog_snapshot(self):
        with self.watchdog_lock:
            return self.watchdog_active

    def response_context(self, response_id):
        if response_id is None:
            return ()
        with self.response_history_lock:
            stored = self.response_history.get(response_id)
            if stored is None:
                raise APIError(404, f"The response `{response_id}` does not exist.",
                               "previous_response_id", "response_not_found")
            return list(stored[0])

    def remember_response(self, response_id, messages):
        saved = list(messages)
        size = len(json.dumps(saved, ensure_ascii=False, separators=(",", ":")).encode())
        with self.response_history_lock:
            old = self.response_history.pop(response_id, None)
            if old:
                self.response_history_bytes -= old[1]
            self.response_history[response_id] = (saved, size)
            self.response_history_bytes += size
            self.response_history.move_to_end(response_id)
            while (len(self.response_history) > RESPONSE_HISTORY_ENTRIES or
                   self.response_history_bytes > RESPONSE_HISTORY_BYTES):
                _key, (_messages, removed) = self.response_history.popitem(last=False)
                self.response_history_bytes -= removed


class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "colibri"

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] %s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, status, body, request_id=None, headers=None):
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if request_id:
            self.send_header("x-request-id", request_id)
            self.send_header("request-id", request_id)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def send_api_error(self, error, request_id, path):
        if path.startswith("/v1/messages"):
            body = anthropic_error_object(error, request_id)
        elif path.startswith("/api/"):
            body = ollama_error_object(error)
        else:
            body = error_object(error)
        self.send_json(error.status, body, request_id, error.headers)

    def send_cors_headers(self):
        origin = self.headers.get("Origin")
        if not origin or ("*" not in self.server.cors_origins and origin not in self.server.cors_origins):
            return
        self.send_header("Access-Control-Allow-Origin", "*" if "*" in self.server.cors_origins else origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta")
        self.send_header("Access-Control-Expose-Headers",
                         "x-request-id, x-colibri-queue-wait-ms, Retry-After")
        self.send_header("Access-Control-Max-Age", "600")
        if "*" not in self.server.cors_origins:
            self.send_header("Vary", "Origin")

    def require_auth(self):
        if self.server.api_key:
            import hmac
            if hmac.compare_digest(self.headers.get("x-api-key", ""), self.server.api_key):
                return
            provided = self.headers.get("Authorization", "")
            expected = f"Bearer {self.server.api_key}"
            if not hmac.compare_digest(provided, expected):
                raise APIError(401, "Invalid or missing API key.", None, "invalid_api_key",
                               "authentication_error")

    def validate_prompt(self, prompt, body):
        if not self.server.context_length:
            return
        maximum, _temperature, _top_p = generation_options(body, self.server.max_tokens)
        input_limit = max(1, self.server.context_length - maximum)
        if len(prompt.encode()) > input_limit * PROMPT_BYTES_PER_TOKEN_LIMIT:
            raise APIError(400, "Rendered prompt exceeds the configured context window.",
                           "messages", "context_length_exceeded")

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError(400, "Invalid Content-Length header.")
        if length < 1 or length > MAX_BODY:
            raise APIError(400, f"Request body must be between 1 and {MAX_BODY} bytes.")
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise APIError(400, "Request body must be valid JSON.")
        if not isinstance(body, dict):
            raise APIError(400, "Request body must be a JSON object.")
        return body

    def check_model(self, body):
        model = body.get("model")
        if model not in self.server.accepted_model_ids:
            raise APIError(404, f"The model `{model}` does not exist.", "model", "model_not_found")

    WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

    def serve_static(self, path):
        """Serve the built web UI (web/dist) so `coli web` is one process.
        Read-only, no auth (same trust level as /health), traversal-safe."""
        if path.startswith("/v1/") or path.startswith("/api/") or path == "/health":
            return False
        base = self.WEB_DIST.resolve()
        if not base.is_dir():
            return False
        rel = unquote(path).lstrip("/") or "index.html"
        target = (base / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            target = None
        if target is None or not target.is_file():
            if path == "/" or "." not in rel:      # SPA fallback
                target = base / "index.html"
                if not target.is_file():
                    return False
            else:
                return False
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)
        return True

    def do_GET(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            path = urlsplit(self.path).path
            if path == "/health":
                payload = {"status": "ok", "scheduler": self.server.scheduler.snapshot(),
                           "kv_slots": self.server.kv_slots,
                           "watchdog_active": self.server.watchdog_snapshot()}
                tiers = getattr(self.server.engine, "tiers", None) if self.server.engine else None
                if tiers: payload["tiers"] = tiers
                hwinfo = getattr(self.server.engine, "hwinfo", None) if self.server.engine else None
                if hwinfo: payload["hwinfo"] = hwinfo
                self.send_json(200, payload, request_id)
                return
            if path == "/experts":
                eng = self.server.engine
                payload = {"rows": 0, "cols": 0, "map": "", "hits": "", "seq": 0}
                if eng and getattr(eng, "emap", None):
                    payload.update(eng.emap)
                    payload["hits"] = eng.hits or ""
                    payload["seq"] = eng.hits_seq
                self.send_json(200, payload, request_id)
                return
            if path == "/profile":
                self.require_auth()
                eng = self.server.engine
                payload = {"seq": getattr(eng, "profile_seq", 0) if eng else 0,
                           "turns": list(getattr(eng, "profile", ()) or ()) if eng else []}
                self.send_json(200, payload, request_id)
                return
            if self.serve_static(path):
                return
            self.require_auth()
            if path == "/v1/models":
                models = [model_object(
                    model_id, self.server.created, self.server.context_length,
                    self.server.max_tokens)
                    for model_id in self.server.model_ids]
                self.send_json(200, {"object": "list", "data": models,
                                     "has_more": False,
                                     "first_id": models[0]["id"] if models else None,
                                     "last_id": models[-1]["id"] if models else None}, request_id)
            elif path.startswith("/v1/models/") and unquote(path[11:]) in self.server.model_ids:
                self.send_json(200, model_object(unquote(path[11:]), self.server.created,
                                                 self.server.context_length,
                                                 self.server.max_tokens), request_id)
            elif path == "/api/version":
                self.send_json(200, {"version": "colibri-1.0"}, request_id)
            elif path in ("/api/tags", "/api/ps"):
                created_at = datetime.datetime.fromtimestamp(
                    self.server.model_modified or self.server.created,
                    datetime.timezone.utc).isoformat().replace("+00:00", "Z")
                models = [{"name": model_id, "model": model_id,
                           "modified_at": created_at, "size": self.server.model_size,
                           # Ollama's CLI slices the first 12 digest bytes when
                           # rendering `ollama list`; short IDs panic in older
                           # releases.
                           "digest": "sha256:bb43a640f04c8e5504a8fbc8c6980455029f3e8fc1dedff10bcd04f94c4f4319",
                           "details": {
                               "format": "colibri", "family": "glm", "families": ["glm"],
                               "parameter_size": "744B", "quantization_level": "Q4"}}
                          for model_id in self.server.model_ids]
                if path == "/api/ps":
                    for model in models:
                        model["expires_at"] = "9999-12-31T23:59:59Z"
                        model["size_vram"] = 0
                self.send_json(200, {"models": models}, request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_api_error(error, request_id, urlsplit(self.path).path)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        request_id = "req_" + uuid.uuid4().hex
        path = urlsplit(self.path).path
        try:
            self.require_auth()
            body = self.read_json()
            if path not in ("/v1/messages/count_tokens", "/api/show"):
                self.check_model(body)
            if path == "/v1/chat/completions":
                self.chat_completion(body, request_id)
            elif path == "/v1/completions":
                self.completion(body, request_id)
            elif path == "/v1/responses":
                self.responses_api(body, request_id)
            elif path == "/v1/messages":
                self.check_model(body)
                self.anthropic_message(body, request_id)
            elif path == "/v1/messages/count_tokens":
                self.check_model(body)
                self.anthropic_count_tokens(body, request_id)
            elif path == "/api/chat":
                self.ollama_chat(body, request_id)
            elif path == "/api/generate":
                self.ollama_generate(body, request_id)
            elif path == "/api/show":
                self.ollama_show(body, request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_api_error(error, request_id, path)
        except ClientCancelled:
            pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.log_error("request failed: %s", error)
            api_error = APIError(500, "The colibri engine failed to process the request.",
                                 None, "engine_error", "server_error")
            try:
                self.send_api_error(api_error, request_id, path)
            except OSError:
                pass

    def collect_generation(self, body, prompt, thinking):
        maximum, temperature, top_p = generation_options(body, self.server.max_tokens)
        cache_slot = body.get("cache_slot")
        if (cache_slot is not None and
                (isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or
                 not 0 <= cache_slot < self.server.kv_slots)):
            raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                           "cache_slot")
        output = []
        watchdog = self.headers.get("X-Colibri-Watchdog") == "1"
        with self.server.watchdog_request(watchdog), \
                self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            stats = self.server.engine.generate(
                prompt, maximum, temperature, top_p, output.append, cache_slot,
                self.client_disconnected)
        reasoning, text = split_reasoning("".join(output), thinking)
        return reasoning, text, stats, {"x-colibri-queue-wait-ms": str(round(queue_wait * 1000))}

    def protocol_stream(self, body, prompt, request_id, consumer_factory, finish, fail):
        maximum, temperature, top_p = generation_options(body, self.server.max_tokens)
        cache_slot = body.get("cache_slot")
        if (cache_slot is not None and
                (isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or
                 not 0 <= cache_slot < self.server.kv_slots)):
            raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                           "cache_slot")
        watchdog = self.headers.get("X-Colibri-Watchdog") == "1"
        with self.server.watchdog_request(watchdog), \
                self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("x-request-id", request_id)
            self.send_header("x-colibri-queue-wait-ms", str(round(queue_wait * 1000)))
            self.send_cors_headers()
            self.end_headers()
            connected = True
            lock = threading.Lock()
            last_write = [time.time()]
            stop = threading.Event()

            def write(data):
                nonlocal connected
                if not connected:
                    return
                with lock:
                    try:
                        self.wfile.write(data)
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected = False

            def keepalive():
                while not stop.wait(1):
                    if not connected:
                        return
                    if time.time() - last_write[0] >= 10:
                        write(b": keep-alive\n\n")

            thread = threading.Thread(target=keepalive, daemon=True)
            thread.start()
            consumer = None
            try:
                consumer = consumer_factory(write)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, consumer.feed, cache_slot,
                    lambda: not connected)
                consumer.close()
                consumer = None
                finish(write, stats)
            except ClientCancelled:
                raise
            except Exception as error:
                self.log_error("stream generation failed: %s", error)
                if consumer is not None:
                    consumer.close()
                    consumer = None
                fail(write, error)
            finally:
                if consumer is not None:
                    consumer.close()
                stop.set()
                thread.join(timeout=2)
                self.close_connection = True

    def responses_api(self, body, request_id):
        normalized = dict(body)
        previous = self.server.response_context(body.get("previous_response_id"))
        normalized["messages"] = responses_messages(body, previous)
        if "max_output_tokens" in body:
            normalized["max_tokens"] = body["max_output_tokens"]
        thinking, effort = reasoning_settings(normalized, self.server.default_thinking)
        prompt = render_chat(normalized["messages"], thinking, effort,
                             normalized.get("tools"), normalized.get("tool_choice"))
        self.validate_prompt(prompt, normalized)
        response_id = "resp_" + uuid.uuid4().hex[:24]
        message_id = "msg_" + uuid.uuid4().hex[:24]
        created = int(time.time())
        model = body["model"]
        tools = normalized.get("tools") or []

        if not body.get("stream", False):
            reasoning, text, stats, headers = self.collect_generation(normalized, prompt, thinking)
            content_text_value, calls = parse_tool_calls(text, tools) if tools else (text, [])
            output = []
            if content_text_value or not calls:
                content = [{"type": "output_text", "text": content_text_value, "annotations": []}]
                output.append({"id": message_id, "type": "message", "status": "completed",
                               "role": "assistant", "content": content})
            for call in calls:
                output.append({"id": "fc_" + uuid.uuid4().hex[:24], "type": "function_call",
                               "status": "completed", "call_id": call["id"],
                               "name": call["function"]["name"],
                               "arguments": call["function"]["arguments"]})
            assistant = {"role": "assistant", "content": content_text_value or None}
            if calls:
                assistant["tool_calls"] = calls
            self.server.remember_response(response_id, [*normalized["messages"], assistant])
            usage = {"input_tokens": stats["prompt_tokens"],
                     "input_tokens_details": {"cached_tokens": 0},
                     "output_tokens": stats["completion_tokens"],
                     "output_tokens_details": {"reasoning_tokens": 0},
                     "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"]}
            self.send_json(200, {"id": response_id, "object": "response",
                "created_at": created, "status": "completed", "model": model,
                "output": output, "usage": usage}, request_id, headers)
            return

        sequence = [0]
        text = []
        raw_tool_text = []
        def event(write, payload):
            payload["sequence_number"] = sequence[0]
            sequence[0] += 1
            write(("data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                   + "\n\n").encode())
        def factory(write):
            event(write, {"type": "response.created", "response": {
                "id": response_id, "object": "response", "created_at": created,
                "status": "in_progress", "model": model, "output": []}})
            if not tools:
                item = {"id": message_id, "type": "message", "status": "in_progress",
                        "role": "assistant", "content": []}
                event(write, {"type": "response.output_item.added",
                              "output_index": 0, "item": item})
                event(write, {"type": "response.content_part.added", "item_id": message_id,
                              "output_index": 0, "content_index": 0,
                              "part": {"type": "output_text", "text": "",
                                       "annotations": []}})
            def on_content(chunk):
                if tools:
                    raw_tool_text.append(chunk)
                    return
                text.append(chunk)
                event(write, {"type": "response.output_text.delta", "item_id": message_id,
                              "output_index": 0, "content_index": 0, "delta": chunk})
            return ReasoningStream(thinking, lambda _chunk: None, on_content)
        def done(write, stats):
            calls = []
            if tools:
                value, calls = parse_tool_calls("".join(raw_tool_text), tools)
                if value:
                    text.append(value)
                    item = {"id": message_id, "type": "message", "status": "in_progress",
                            "role": "assistant", "content": []}
                    event(write, {"type": "response.output_item.added",
                                  "output_index": 0, "item": item})
                    event(write, {"type": "response.content_part.added", "item_id": message_id,
                                  "output_index": 0, "content_index": 0,
                                  "part": {"type": "output_text", "text": "",
                                           "annotations": []}})
                    event(write, {"type": "response.output_text.delta", "item_id": message_id,
                                  "output_index": 0, "content_index": 0, "delta": value})
            output = []
            if text or not calls:
                value = "".join(text)
                part = {"type": "output_text", "text": value, "annotations": []}
                event(write, {"type": "response.output_text.done", "item_id": message_id,
                              "output_index": 0, "content_index": 0, "text": value})
                event(write, {"type": "response.content_part.done", "item_id": message_id,
                              "output_index": 0, "content_index": 0, "part": part})
                item = {"id": message_id, "type": "message", "status": "completed",
                        "role": "assistant", "content": [part]}
                event(write, {"type": "response.output_item.done",
                              "output_index": 0, "item": item})
                output.append(item)
            for call in calls:
                index = len(output)
                item = {"id": "fc_" + uuid.uuid4().hex[:24], "type": "function_call",
                        "status": "in_progress", "call_id": call["id"],
                        "name": call["function"]["name"], "arguments": ""}
                event(write, {"type": "response.output_item.added",
                              "output_index": index, "item": item})
                arguments = call["function"]["arguments"]
                event(write, {"type": "response.function_call_arguments.delta",
                              "item_id": item["id"], "output_index": index,
                              "delta": arguments})
                event(write, {"type": "response.function_call_arguments.done",
                              "item_id": item["id"], "output_index": index,
                              "arguments": arguments})
                item["status"] = "completed"
                item["arguments"] = arguments
                event(write, {"type": "response.output_item.done",
                              "output_index": index, "item": item})
                output.append(item)
            assistant = {"role": "assistant", "content": "".join(text) or None}
            if calls:
                assistant["tool_calls"] = calls
            self.server.remember_response(response_id, [*normalized["messages"], assistant])
            usage = {"input_tokens": stats["prompt_tokens"],
                     "input_tokens_details": {"cached_tokens": 0},
                     "output_tokens": stats["completion_tokens"],
                     "output_tokens_details": {"reasoning_tokens": 0},
                     "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"]}
            event(write, {"type": "response.completed", "response": {
                "id": response_id, "object": "response", "created_at": created,
                "status": "completed", "model": model, "output": output, "usage": usage}})
        def failed(write, _error):
            event(write, {"type": "response.failed", "response": {
                "id": response_id, "object": "response", "created_at": created,
                "status": "failed", "model": model,
                "error": {"code": "engine_error",
                          "message": "The colibri engine failed to process the request."}}})
        self.protocol_stream(normalized, prompt, request_id, factory, done, failed)

    def anthropic_count_tokens(self, body, request_id):
        normalized = dict(body)
        normalized["messages"] = anthropic_messages(body)
        normalized["tools"] = anthropic_tools(body.get("tools"))
        thinking, effort = reasoning_settings(normalized, self.server.default_thinking)
        prompt = render_chat(normalized["messages"], thinking, effort,
                             normalized["tools"], normalized.get("tool_choice"))
        estimate = max(1, math.ceil(len(prompt.encode("utf-8")) / 4))
        self.send_json(200, {"input_tokens": estimate}, request_id,
                       {"x-colibri-token-count": "estimated"})

    @staticmethod
    def _ollama_time():
        return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    def ollama_show(self, body, request_id):
        model = body.get("model") or body.get("name")
        if model not in self.server.accepted_model_ids:
            raise APIError(404, f"The model `{model}` does not exist.", "model", "model_not_found")
        self.send_json(200, {"modelfile": "FROM colibri\n", "parameters": "",
            "template": "GLM-5.2 native chat template", "details": {
                "parent_model": "", "format": "colibri", "family": "glm",
                "families": ["glm"], "parameter_size": "744B", "quantization_level": "Q4"},
            "model_info": {"general.architecture": "glm4moe",
                           "general.parameter_count": 744000000000,
                           "glm.context_length": self.server.context_length or 0},
            # Ollama automatically enables `think` when this endpoint advertises
            # thinking.  GLM can finish directly without closing </think>, which
            # makes Ollama display the answer as reasoning and can induce long
            # repetitions.  Explicit API requests may still opt into `think`.
            "capabilities": ["completion", "tools"]}, request_id)

    def _ollama_run(self, body, prompt, request_id, chat, tools, thinking):
        _normalized, maximum, temperature, top_p = ollama_options(body, self.server.max_tokens)
        stream = body.get("stream", True)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        cache_slot = body.get("cache_slot")
        if cache_slot is not None and (isinstance(cache_slot, bool) or
                not isinstance(cache_slot, int) or not 0 <= cache_slot < self.server.kv_slots):
            raise APIError(400, "Invalid cache slot.", "cache_slot")
        model, started = body["model"], time.monotonic_ns()
        watchdog = self.headers.get("X-Colibri-Watchdog") == "1"
        with self.server.watchdog_request(watchdog), \
                self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            headers = {"x-colibri-queue-wait-ms": str(round(queue_wait * 1000))}
            output = []
            if not stream:
                stats = self.server.engine.generate(prompt, maximum, temperature, top_p,
                                                    output.append, cache_slot,
                                                    self.client_disconnected)
                reasoning, text = split_reasoning("".join(output), thinking)
                text, calls = parse_tool_calls(text, tools) if tools else (text, [])
                elapsed = time.monotonic_ns() - started
                result = {"model": model, "created_at": self._ollama_time(), "done": True,
                          "done_reason": "length" if stats["length_limited"] else "stop",
                          "total_duration": elapsed, "load_duration": 0,
                          "prompt_eval_count": stats["prompt_tokens"], "prompt_eval_duration": 0,
                          "eval_count": stats["completion_tokens"], "eval_duration": elapsed}
                if chat:
                    message = {"role": "assistant", "content": text}
                    if reasoning: message["thinking"] = reasoning
                    if calls:
                        message["tool_calls"] = [{"function": {
                            "name": call["function"]["name"],
                            "arguments": json.loads(call["function"]["arguments"])}}
                            for call in calls]
                    result["message"] = message
                else:
                    result["response"] = text
                    if reasoning: result["thinking"] = reasoning
                self.send_json(200, result, request_id, headers)
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("x-request-id", request_id)
            for name, value in headers.items(): self.send_header(name, value)
            self.send_cors_headers(); self.end_headers()
            connected, raw = True, []
            def write(payload):
                nonlocal connected
                if not connected: return
                try:
                    self.wfile.write((json.dumps(payload, ensure_ascii=False,
                                                 separators=(",", ":")) + "\n").encode())
                    self.wfile.flush()
                except OSError:
                    connected = False
            def emit(kind, chunk):
                payload = {"model": model, "created_at": self._ollama_time(), "done": False}
                if chat:
                    payload["message"] = {"role": "assistant", "content": ""}
                    payload["message"][kind] = chunk
                else:
                    payload.update({"response": "", kind: chunk})
                write(payload)
            def content(chunk):
                raw.append(chunk)
                if not tools: emit("content" if chat else "response", chunk)
            splitter = ReasoningStream(thinking, lambda chunk: emit("thinking", chunk), content)
            stats = self.server.engine.generate(prompt, maximum, temperature, top_p,
                                                splitter.feed, cache_slot, lambda: not connected)
            splitter.close()
            if tools:
                text, calls = parse_tool_calls("".join(raw), tools)
                if text: emit("content", text)
                for call in calls:
                    write({"model": model, "created_at": self._ollama_time(), "done": False,
                           "message": {"role": "assistant", "content": "", "tool_calls": [{
                               "function": {"name": call["function"]["name"], "arguments":
                                            json.loads(call["function"]["arguments"])}}]}})
            elapsed = time.monotonic_ns() - started
            final = {"model": model, "created_at": self._ollama_time(), "done": True,
                     "done_reason": "length" if stats["length_limited"] else "stop",
                     "total_duration": elapsed, "load_duration": 0,
                     "prompt_eval_count": stats["prompt_tokens"], "prompt_eval_duration": 0,
                     "eval_count": stats["completion_tokens"], "eval_duration": elapsed}
            final["message" if chat else "response"] = ({"role": "assistant", "content": ""}
                                                         if chat else "")
            write(final)
            self.close_connection = True

    def ollama_chat(self, body, request_id):
        messages = ollama_messages(body.get("messages"))
        tools = body.get("tools") or None
        if tools is not None and not isinstance(tools, list):
            raise APIError(400, "`tools` must be an array.", "tools")
        normalized = dict(body)
        if isinstance(body.get("think"), str):
            normalized["reasoning_effort"] = body["think"]; normalized.pop("think")
        thinking, effort = reasoning_settings(normalized, self.server.default_thinking)
        prompt = render_chat(messages, thinking, effort, tools, body.get("tool_choice"))
        self.validate_prompt(prompt, {**body, "max_tokens": ollama_options(
            body, self.server.max_tokens)[1]})
        self._ollama_run(body, prompt, request_id, True, tools, thinking)

    def ollama_generate(self, body, request_id):
        prompt = body.get("prompt", "")
        if not isinstance(prompt, str):
            raise APIError(400, "`prompt` must be a string.", "prompt")
        if body.get("images"):
            raise APIError(400, "GLM-5.2 Colibri is text-only; images are unsupported.",
                           "images", "unsupported_content_type")
        normalized = dict(body)
        if isinstance(body.get("think"), str):
            normalized["reasoning_effort"] = body["think"]; normalized.pop("think")
        thinking, effort = reasoning_settings(normalized, self.server.default_thinking)
        if not body.get("raw", False):
            messages = []
            if body.get("system") is not None:
                messages.append({"role": "system", "content": body["system"]})
            messages.append({"role": "user", "content": prompt})
            prompt = render_chat(messages, thinking, effort)
        self.validate_prompt(prompt, {**body, "max_tokens": ollama_options(
            body, self.server.max_tokens)[1]})
        self._ollama_run(body, prompt, request_id, False, None, thinking)

    def anthropic_message(self, body, request_id):
        if "max_tokens" not in body:
            raise APIError(400, "`max_tokens` is required.", "max_tokens")
        normalized = dict(body)
        if isinstance(normalized.get("thinking"), dict) and \
                normalized["thinking"].get("type") == "adaptive":
            normalized["thinking"] = {**normalized["thinking"], "type": "enabled"}
        normalized["messages"] = anthropic_messages(body)
        normalized["tools"] = anthropic_tools(body.get("tools"))
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            choice_type = choice.get("type")
            if choice_type == "auto":
                normalized["tool_choice"] = "auto"
            elif choice_type == "any":
                normalized["tool_choice"] = "required"
            elif choice_type == "none":
                normalized["tool_choice"] = "none"
            elif choice_type == "tool" and isinstance(choice.get("name"), str):
                normalized["tool_choice"] = {"function": {"name": choice["name"]}}
            else:
                raise APIError(400, "Unsupported Anthropic `tool_choice`.", "tool_choice")
        thinking, effort = reasoning_settings(normalized, self.server.default_thinking)
        prompt = render_chat(normalized["messages"], thinking, effort,
                             normalized["tools"], normalized.get("tool_choice"))
        self.validate_prompt(prompt, normalized)
        completion_id = "msg_" + uuid.uuid4().hex[:24]
        model = body["model"]
        tools = normalized["tools"]

        if not body.get("stream", False):
            reasoning, text, stats, headers = self.collect_generation(normalized, prompt, thinking)
            text, calls = parse_tool_calls(text, tools) if tools else (text, [])
            content = []
            if reasoning:
                content.append({"type": "thinking", "thinking": reasoning,
                                "signature": completion_id})
            if text or not calls:
                content.append({"type": "text", "text": text})
            for call in calls:
                content.append({"type": "tool_use", "id": call["id"],
                                "name": call["function"]["name"],
                                "input": json.loads(call["function"]["arguments"])})
            self.send_json(200, {"id": completion_id, "type": "message",
                "role": "assistant", "model": model, "content": content,
                "stop_reason": ("tool_use" if calls else
                                "max_tokens" if stats["length_limited"] else "end_turn"),
                "stop_sequence": None,
                "usage": {"input_tokens": stats["prompt_tokens"],
                          "output_tokens": stats["completion_tokens"],
                          "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0}},
                request_id, headers)
            return

        state = {"index": -1, "kind": None}
        raw_tool_text = []
        def event(write, name, payload):
            write((f"event: {name}\ndata: "
                   + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                   + "\n\n").encode())
        def factory(write):
            event(write, "message_start", {"type": "message_start", "message": {
                "id": completion_id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}})
            def start(kind, block=None):
                if state["kind"] is not None:
                    event(write, "content_block_stop",
                          {"type": "content_block_stop", "index": state["index"]})
                state["index"] += 1
                state["kind"] = kind
                if block is None:
                    block = ({"type": "thinking", "thinking": "", "signature": ""}
                             if kind == "thinking" else {"type": "text", "text": ""})
                event(write, "content_block_start", {"type": "content_block_start",
                                                     "index": state["index"],
                                                     "content_block": block})
            state["start"] = start
            def reasoning(chunk):
                if state["kind"] != "thinking":
                    start("thinking")
                event(write, "content_block_delta", {"type": "content_block_delta",
                    "index": state["index"], "delta": {"type": "thinking_delta",
                                                       "thinking": chunk}})
            def content(chunk):
                if tools:
                    raw_tool_text.append(chunk)
                    return
                if state["kind"] != "text":
                    start("text")
                event(write, "content_block_delta", {"type": "content_block_delta",
                    "index": state["index"], "delta": {"type": "text_delta", "text": chunk}})
            return ReasoningStream(thinking, reasoning, content)
        def done(write, stats):
            calls = []
            if tools:
                text, calls = parse_tool_calls("".join(raw_tool_text), tools)
                if text:
                    state["start"]("text")
                    event(write, "content_block_delta", {"type": "content_block_delta",
                        "index": state["index"], "delta": {"type": "text_delta",
                                                           "text": text}})
                for call in calls:
                    state["start"]("tool_use", {"type": "tool_use", "id": call["id"],
                        "name": call["function"]["name"], "input": {}})
                    event(write, "content_block_delta", {"type": "content_block_delta",
                        "index": state["index"], "delta": {"type": "input_json_delta",
                        "partial_json": call["function"]["arguments"]}})
            if state["kind"] is not None:
                event(write, "content_block_stop",
                      {"type": "content_block_stop", "index": state["index"]})
            event(write, "message_delta", {"type": "message_delta",
                "delta": {"stop_reason": ("tool_use" if calls else
                                         "max_tokens" if stats["length_limited"] else "end_turn"),
                          "stop_sequence": None},
                "usage": {"input_tokens": stats["prompt_tokens"],
                          "output_tokens": stats["completion_tokens"],
                          "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0}})
            event(write, "message_stop", {"type": "message_stop"})
        def failed(write, _error):
            event(write, "error", {"type": "error", "error": {
                "type": "api_error",
                "message": "The colibri engine failed to process the request."}})
        self.protocol_stream(normalized, prompt, request_id, factory, done, failed)

    def generation(self, body, prompt, request_id, chat, thinking=False):
        # COLI_DEBUG tees the engine transaction to stderr: 1 = decoded output stream only,
        # 2 = both sides (rendered prompt + output). render_chat already folds prior turns and
        # tool results into `prompt`, so level 2 is the full conversation the engine saw.
        try:
            dbg = int(os.environ.get("COLI_DEBUG", "0"))
        except ValueError:
            dbg = 0
        if dbg >= 2:
            sys.stderr.write(f"\n===== PROMPT [{request_id}] =====\n{prompt}\n===== OUTPUT [{request_id}] =====\n")
            sys.stderr.flush()
        maximum, temperature, top_p = generation_options(body, self.server.max_tokens)
        tools = (body.get("tools") or body.get("functions") or None) if chat else None
        if body.get("tool_choice") == "none":
            tools = None          # client forbade tools: never surface tool_calls
        cache_slot = body.get("cache_slot")
        if (cache_slot is not None and
                (isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or
                 not 0 <= cache_slot < self.server.kv_slots)):
            raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                           "cache_slot")
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        stream_options = body.get("stream_options") if stream else None
        if stream and stream_options is not None and not isinstance(stream_options, dict):
            raise APIError(400, "`stream_options` must be an object.", "stream_options")
        include_usage = bool((stream_options or {}).get("include_usage"))
        object_name = "chat.completion" if chat else "text_completion"
        id_prefix = "chatcmpl-" if chat else "cmpl-"
        completion_id = id_prefix + uuid.uuid4().hex
        created = int(time.time())
        response_model = body.get("model", self.server.model_id)

        watchdog = self.headers.get("X-Colibri-Watchdog") == "1"
        with self.server.watchdog_request(watchdog), \
                self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            queue_headers = {"x-colibri-queue-wait-ms": str(round(queue_wait * 1000))}
            if not stream:
                output = []
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, output.append, cache_slot,
                    self.client_disconnected)
                reasoning, text = split_reasoning("".join(output), thinking)
                length_finish = "length" if stats["length_limited"] else "stop"
                if chat and tools:
                    content, calls = parse_tool_calls(text, tools)
                    message = {"role": "assistant", "content": content or None, "refusal": None}
                    if reasoning:
                        message["reasoning_content"] = reasoning
                    if calls:
                        message["tool_calls"] = calls
                    finish = "tool_calls" if calls else length_finish
                    choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                else:
                    message = {"role": "assistant", "content": text, "refusal": None}
                    if reasoning:
                        message["reasoning_content"] = reasoning
                    choice = ({"index": 0, "message": message,
                               "logprobs": None, "finish_reason": length_finish} if chat else
                              {"index": 0, "text": text, "logprobs": None, "finish_reason": length_finish})
                self.send_json(200, {"id": completion_id, "object": object_name, "created": created,
                    "model": response_model, "choices": [choice], "usage": self.usage(stats)},
                    request_id, queue_headers)
                return

            stream_object = "chat.completion.chunk" if chat else object_name
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("x-request-id", request_id)
            for name, value in queue_headers.items(): self.send_header(name, value)
            self.send_cors_headers()
            self.end_headers()
            connected = True
            # KEEPALIVE: engine.generate() blocks SILENTLY during the (minutes-long) cold
            # prefill, and the client drops the socket after its idle timeout. A background pump
            # emits a reasoning_content "." delta (the channel that reliably resets the client's
            # timer and lands in the thinking panel, so answer content stays clean) whenever no
            # event has been written for KA_GAP seconds. All wfile writes share ka_lock so the
            # pump and event() never interleave; last_write gates the pump so it stays quiet
            # while real tokens are flowing (e.g. during decode).
            ka_lock = threading.Lock()
            last_write = [time.time()]
            ka_stop = threading.Event()
            KA_GAP = 10.0
            dbg_echo = dbg >= 1   # tee decoded tokens to stderr (COLI_DEBUG level parsed in generation())

            def event(choices, usage_marker=False):
                nonlocal connected
                if not connected:
                    return
                event_body = {"id": completion_id, "object": stream_object, "created": created,
                              "model": response_model, "choices": choices}
                if include_usage:
                    event_body["usage"] = None if not usage_marker else usage_marker
                data = json.dumps(event_body, ensure_ascii=False, separators=(",", ":"))
                with ka_lock:
                    try:
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected = False

            def _keepalive():
                while not ka_stop.wait(1.0):
                    if not connected:
                        return
                    if time.time() - last_write[0] >= KA_GAP:
                        with ka_lock:
                            try:
                                self.wfile.write(b": keep-alive\n\n")
                                self.wfile.flush()
                                last_write[0] = time.time()
                            except OSError:
                                connected = False

            if chat:
                event([{"index": 0, "delta": {"role": "assistant", "content": ""},
                        "logprobs": None, "finish_reason": None}])

            def emit_content(text):
                choice = ({"index": 0, "delta": {"content": text}, "logprobs": None,
                           "finish_reason": None} if chat else
                          {"index": 0, "text": text, "logprobs": None, "finish_reason": None})
                event([choice])

            def emit_reasoning(text):
                event([{"index": 0, "delta": {"reasoning_content": text}, "logprobs": None,
                        "finish_reason": None}])

            ka_thread = threading.Thread(target=_keepalive, daemon=True)
            ka_thread.start()
            if chat and tools:
                # Suppress tool-call markers from the streamed content and parse the authoritative
                # calls from the FULL reply after generation. Hold back a marker-length tail so a
                # <tool_call> split across engine chunks is still caught.
                sp = {"buf": "", "tool": False}
                hold = len(BOX_START) - 1
                raw = []
                def emit_tools(chunk):
                    raw.append(chunk)
                    if sp["tool"]:
                        return
                    sp["buf"] += chunk
                    cut = sp["buf"].find(BOX_START)
                    if cut >= 0:
                        if cut:
                            emit_content(sp["buf"][:cut])
                        sp["buf"] = ""
                        sp["tool"] = True
                        return
                    flush = max(0, len(sp["buf"]) - hold)
                    if flush:
                        emit_content(sp["buf"][:flush])
                        sp["buf"] = sp["buf"][flush:]
                splitter = ReasoningStream(thinking, emit_reasoning, emit_tools)
                def emit_tools_raw(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    splitter.feed(chunk)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, emit_tools_raw, cache_slot,
                    lambda: not connected)
                splitter.close()
                if not sp["tool"] and sp["buf"]:
                    emit_content(sp["buf"])             # no tool call happened: flush held tail
                _content, calls = parse_tool_calls("".join(raw), tools)
                for i, tc in enumerate(calls):
                    event([{"index": 0, "delta": {"tool_calls": [{"index": i, "id": tc["id"],
                             "type": "function", "function": {"name": tc["function"]["name"],
                             "arguments": tc["function"]["arguments"]}}]},
                            "logprobs": None, "finish_reason": None}])
                finish = "tool_calls" if calls else ("length" if stats["length_limited"] else "stop")
            else:
                def emit_plain(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    splitter.feed(chunk)
                splitter = ReasoningStream(thinking, emit_reasoning, emit_content)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, emit_plain, cache_slot,
                    lambda: not connected)
                splitter.close()
                finish = "length" if stats["length_limited"] else "stop"
            ka_stop.set()                          # generation done: stop the keepalive pump
            ka_thread.join(timeout=2)
            final_choice = ({"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish}
                            if chat else {"index": 0, "text": "", "logprobs": None,
                                          "finish_reason": finish})
            event([final_choice])
            if include_usage:
                event([], self.usage(stats))
            if connected:
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
            self.close_connection = True

    def client_disconnected(self):
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
            return self.connection.recv(1, flags) == b""
        except (OSError, ValueError):
            return True

    @staticmethod
    def usage(stats):
        prompt = stats["prompt_tokens"]
        completion = stats["completion_tokens"]
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion}

    def chat_completion(self, body, request_id):
        enable_thinking, reasoning_effort = reasoning_settings(
            body, self.server.default_thinking or os.environ.get("COLI_THINK", "0") == "1")
        tools = body.get("tools") or body.get("functions") or None
        prompt = render_chat(body.get("messages"), enable_thinking, reasoning_effort, tools,
                             body.get("tool_choice"))
        self.validate_prompt(prompt, body)
        self.generation(body, prompt, request_id, True, enable_thinking)

    def completion(self, body, request_id):
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise APIError(400, "Colibri currently requires `prompt` to be a string.", "prompt")
        if not prompt:
            raise APIError(400, "`prompt` must not be empty.", "prompt")
        self.validate_prompt(prompt, body)
        self.generation(body, prompt, request_id, False)


def serve(model, host="127.0.0.1", port=8000, model_id="glm-5.2", api_key=None,
          cap=8, max_tokens=1024, engine=HERE / "glm", env=None, cors_origins=None,
          max_queue=8, queue_timeout=300, kv_slots=1, expert_bits=8, dense_bits=None,
          model_aliases=(), hidden_model_aliases=(), context_length=None,
          default_thinking=False):
    if not 1 <= max_tokens:
        raise ValueError("max_tokens must be positive")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if max_queue < 0:
        raise ValueError("max_queue cannot be negative")
    if queue_timeout <= 0:
        raise ValueError("queue_timeout must be positive")
    if not 1 <= kv_slots <= 16:
        raise ValueError("kv_slots must be between 1 and 16")
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        print("WARNING: API is listening beyond localhost without COLI_API_KEY", file=sys.stderr)
    origins = DEFAULT_CORS_ORIGINS if cors_origins is None else tuple(cors_origins)
    if context_length is None:
        context_length = int((env or os.environ).get("CTX", "0")) or None
    # Bind before starting the 744B engine. A stale/occupied port must fail in
    # milliseconds rather than loading hundreds of GB and leaking a child.
    server = APIServer((host, port), None, model_id, api_key, max_tokens, origins,
                       max_queue, queue_timeout, kv_slots, model_aliases,
                       hidden_model_aliases, context_length, default_thinking, model)
    runtime = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        runtime = Engine(engine, model, cap, max_tokens, env, kv_slots,
                         expert_bits, dense_bits)
        server.engine = runtime
        print(f"OpenAI-compatible API listening on http://{host}:{port}/v1", file=sys.stderr)
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.scheduler.close()
        server.server_close()
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("COLI_MODEL"), required=not os.environ.get("COLI_MODEL"))
    parser.add_argument("--engine", default=str(HERE / "glm"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=os.environ.get("COLI_MODEL_ID", "glm-5.2"))
    parser.add_argument("--model-alias", action="append", default=[])
    parser.add_argument("--hidden-model-alias", action="append", default=[])
    parser.add_argument("--default-thinking", action="store_true",
                        default=os.environ.get("COLI_THINK", "0") == "1")
    parser.add_argument("--context-length", type=int,
                        default=int(os.environ.get("COLI_CONTEXT_LENGTH", "0")) or None)
    parser.add_argument("--api-key", default=os.environ.get("COLI_API_KEY"))
    parser.add_argument("--cors-origin", action="append", default=None,
                        help="allowed browser origin; repeat as needed (use '*' for any origin)")
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--expert-bits", type=int,
                        default=int(os.environ.get("COLI_EXPERT_BITS", "8")))
    parser.add_argument("--dense-bits", type=int,
                        default=int(os.environ["COLI_DENSE_BITS"])
                        if os.environ.get("COLI_DENSE_BITS") else None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-queue", type=int, default=int(os.environ.get("COLI_MAX_QUEUE", "8")))
    parser.add_argument("--queue-timeout", type=float,
                        default=float(os.environ.get("COLI_QUEUE_TIMEOUT", "300")))
    parser.add_argument("--kv-slots", type=int, default=int(os.environ.get("COLI_KV_SLOTS", "1")))
    args = parser.parse_args()
    serve(args.model, args.host, args.port, args.model_id, args.api_key,
          args.cap,args.max_tokens,args.engine,cors_origins=args.cors_origin,
          max_queue=args.max_queue,queue_timeout=args.queue_timeout,kv_slots=args.kv_slots,
          expert_bits=args.expert_bits,dense_bits=args.dense_bits,
          model_aliases=args.model_alias,hidden_model_aliases=args.hidden_model_alias,
          context_length=args.context_length,default_thinking=args.default_thinking)


if __name__ == "__main__":
    main()
