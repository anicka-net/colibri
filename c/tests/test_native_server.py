import calendar
import json
import os
import signal
import socket
import struct
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def setUpModule():
    if os.name == "nt":
        raise unittest.SkipTest("native CLI requires POSIX process APIs")
    subprocess.run(["make", "coli-native", "tests/fake_mux_engine"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def write_shard(path, tensors):
    offset, header, payload = 0, {}, b""
    for name, size in tensors:
        header[name] = {"dtype": "U8", "shape": [size],
                        "data_offsets": [offset, offset + size]}
        payload += b"\0" * size
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class NativeServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        cls.port = listener.getsockname()[1]
        listener.close()
        env = dict(os.environ, COLI_ENGINE=str(ROOT / "tests/fake_mux_engine"),
                   OMP_PROC_BIND="false", COLI_ENGINE_OMP_PROC_BIND="spread")
        cls.web_tmp = tempfile.TemporaryDirectory()
        cls.runtime_tmp = tempfile.TemporaryDirectory()
        cls.model_dir = Path(cls.runtime_tmp.name) / "model"
        cls.model_dir.mkdir()
        (cls.model_dir / "weights.bin").write_bytes(b"x" * 123)
        future = int(time.time()) + 86400
        os.utime(cls.model_dir / "weights.bin", (future, future))
        (cls.model_dir / ".coli_usage").write_bytes(b"runtime state")
        cls.model_link = Path(cls.runtime_tmp.name) / "model-link"
        cls.model_link.symlink_to(cls.model_dir, target_is_directory=True)
        web = Path(cls.web_tmp.name)
        (web / "index.html").write_text("native dashboard", encoding="utf-8")
        outside = web.parent / (web.name + "-private")
        outside.mkdir(exist_ok=True)
        (outside / "secret.txt").write_text("private", encoding="utf-8")
        cls.outside = outside
        env["COLI_WEB_ROOT"] = str(web)
        env["XDG_RUNTIME_DIR"] = cls.runtime_tmp.name
        cls.server_env = env
        cls.process = subprocess.Popen([
            str(ROOT / "coli-native"), "serve", "--model", str(cls.model_link),
            "--host", "127.0.0.1", "--port", str(cls.port),
            "--model-id", "glm-test", "--model-alias", "glm-public",
            "--hidden-model-alias", "glm-hidden", "--api-key", "secret",
            "--ngen", "32", "--ctx", "4097", "--kv-slots", "2",
        ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        cls.base = f"http://127.0.0.1:{cls.port}"
        for _ in range(100):
            try:
                cls.request("/health")
                break
            except OSError:
                if cls.process.poll() is not None:
                    raise RuntimeError(cls.process.stderr.read().decode())
                time.sleep(.02)
        else:
            raise RuntimeError("native server did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.process.send_signal(signal.SIGTERM)
        cls.process.wait(timeout=3)
        for child in cls.outside.iterdir():
            child.unlink()
        cls.outside.rmdir()
        cls.web_tmp.cleanup()
        cls.runtime_tmp.cleanup()

    @classmethod
    def request(cls, path, body=None, key="secret"):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {key}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        return urlopen(Request(cls.base + path, data=data, headers=headers), timeout=2)

    def test_auth_health_models_and_profile(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/health", key="wrong")
        self.assertEqual(caught.exception.code, 401)
        with self.request("/health") as response:
            health = json.load(response)
        self.assertEqual(health["kv_slots"], 2)
        self.assertEqual(health["tiers"]["vram"], 2)
        with self.request("/v1/models") as response:
            models = json.load(response)["data"]
        self.assertEqual([m["id"] for m in models], ["glm-test", "glm-public"])
        self.assertEqual(models[0]["context_length"], 4097)

    def test_context_is_propagated_to_engine(self):
        with self.request("/v1/chat/completions", {
            "model": "glm-test", "messages": [{"role": "user",
                                                   "content": "show ctx"}],
        }) as response:
            result = json.load(response)
        self.assertEqual(result["choices"][0]["message"]["content"], "CTX=4097")

    def test_engine_only_openmp_binding_is_propagated(self):
        with self.request("/v1/chat/completions", {
            "model": "glm-test", "messages": [{"role": "user",
                                                   "content": "show bind"}],
        }) as response:
            result = json.load(response)
        self.assertEqual(result["choices"][0]["message"]["content"],
                         "OMP_PROC_BIND=spread")

    def test_anthropic_transport_headers_are_not_rendered_to_model(self):
        body = {
            "model": "glm-test",
            "system": "x-anthropic-billing-header: cc_version=2; cch=random;"
                      "You are Claude Code.",
            "messages": [{"role": "user", "content": "check headers"}],
            "max_tokens": 8,
        }
        with self.request("/v1/messages", body) as response:
            result = json.load(response)
        self.assertEqual(result["content"][0]["text"], "headers-stripped")

    def test_anthropic_authored_system_text_is_preserved(self):
        body = {
            "model": "glm-test",
            "system": [
                {"type": "text",
                 "text": "Authorization: keep this\ncheck authored system"},
                {"type": "text",
                 "text": "x-user-authored: keep this\nsecond block"},
            ],
            "messages": [{"role": "user", "content": "verify system"}],
            "max_tokens": 8,
        }
        with self.request("/v1/messages", body) as response:
            result = json.load(response)
        self.assertEqual(result["content"][0]["text"], "system-preserved")

    def test_private_pidfile_and_stop_dry_run(self):
        pidfile = Path(self.runtime_tmp.name) / f"colibri-serve-{self.port}.pid"
        self.assertTrue(pidfile.is_file())
        self.assertEqual(pidfile.stat().st_mode & 0o777, 0o600)
        run = subprocess.run([
            str(ROOT / "coli-native"), "stop", "--port", str(self.port), "--dry-run",
        ], env=self.server_env, universal_newlines=True, stdout=subprocess.PIPE,
           stderr=subprocess.PIPE, check=True)
        self.assertIn("would stop", run.stdout)

    def test_openai_json_stream_alias_and_cache_slot(self):
        body = {"model": "glm-hidden", "messages": [{"role": "user", "content": "hello"}],
                "cache_slot": 1, "max_tokens": 4}
        with self.request("/v1/chat/completions", body) as response:
            result = json.load(response)
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello from C")
        self.assertEqual(result["usage"]["total_tokens"], 10)
        with self.request("/v1/chat/completions", {**body, "stream": True}) as response:
            stream = response.read().decode()
        self.assertIn('"object":"chat.completion.chunk"', stream)
        self.assertTrue(stream.endswith("data: [DONE]\n\n"))
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {**body, "cache_slot": 2})
        self.assertEqual(caught.exception.code, 400)

    def test_anthropic_responses_and_ollama(self):
        with self.request("/v1/messages", {
            "model": "glm-test", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        }) as response:
            anthropic = json.load(response)
        self.assertEqual(anthropic["type"], "message")
        self.assertEqual(anthropic["content"][0]["text"], "Hello from C")
        with self.request("/v1/responses", {"model": "glm-test", "input": "hello"}) as response:
            responses = json.load(response)
        self.assertEqual(responses["object"], "response")
        self.assertEqual(responses["output"][0]["content"][0]["text"], "Hello from C")
        with self.request("/api/chat", {
            "model": "glm-test", "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }) as response:
            ollama = json.load(response)
        self.assertTrue(ollama["done"])
        self.assertEqual(ollama["message"]["content"], "Hello from C")
        with self.request("/api/tags") as response:
            model = json.load(response)["models"][0]
            self.assertEqual(model["name"], "glm-test")
            self.assertRegex(model["digest"], r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(model["size"], 123)
            modified = calendar.timegm(time.strptime(model["modified_at"],
                                                      "%Y-%m-%dT%H:%M:%SZ"))
            self.assertGreaterEqual(modified, time.time() - 10)
            self.assertLessEqual(modified, time.time() + 1)
        with self.request("/api/show", {"model": "glm-test"}) as response:
            shown = json.load(response)
        self.assertIn("completion", shown["capabilities"])
        self.assertIn("tools", shown["capabilities"])
        self.assertNotIn("thinking", shown["capabilities"])

    def test_profile_collects_engine_telemetry(self):
        with self.request("/v1/completions", {"model": "glm-test", "prompt": "x"}):
            pass
        # PROF follows DONE on the engine stream; allow the dispatcher to consume it.
        time.sleep(.02)
        with self.request("/profile") as response:
            profile = json.load(response)
        self.assertGreaterEqual(profile["seq"], 1)
        self.assertEqual(profile["turns"][-1]["forwards"], 4)

    def test_reasoning_is_separated_in_supported_protocols(self):
        with self.request("/v1/messages", {
            "model": "glm-test", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8, "thinking": {"type": "enabled", "budget_tokens": 4},
        }) as response:
            result = json.load(response)
        self.assertEqual(result["content"][0]["thinking"], "Reasoning")
        self.assertEqual(result["content"][1]["text"], "Answer")
        with self.request("/v1/responses", {
            "model": "glm-test", "input": "hello", "reasoning": {"effort": "high"},
        }) as response:
            result = json.load(response)
        self.assertEqual(result["output"][0]["content"][0]["text"], "Answer")

    def test_cors_preflight(self):
        request = Request(self.base + "/v1/chat/completions", method="OPTIONS", headers={
            "Origin": "http://localhost:5173",
            "Authorization": "Bearer secret",
        })
        with urlopen(request, timeout=2) as response:
            self.assertEqual(response.status, 204)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"],
                             "http://localhost:5173")

    def test_claude_code_query_routing_and_head_probe(self):
        probe = Request(self.base + "/", method="HEAD", headers={
            "Authorization": "Bearer secret",
        })
        with urlopen(probe, timeout=2) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"")

        with self.request("/v1/messages?beta=true", {
            "model": "glm-test", "messages": [{"role": "user",
                                                   "content": "hello"}],
            "max_tokens": 4, "stream": False,
        }) as response:
            result = json.load(response)
        self.assertEqual(result["type"], "message")
        self.assertEqual(result["content"][0]["text"], "Hello from C")

    def test_malformed_json_and_generation_option_types_are_rejected(self):
        malformed = Request(self.base + "/v1/chat/completions", data=b'{"model":',
                            headers={"Authorization": "Bearer secret",
                                     "Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(malformed, timeout=2)
        self.assertEqual(caught.exception.code, 400)

        base = {"model": "glm-test", "messages": [{"role": "user",
                                                      "content": "hello"}]}
        for field, value in (("stream", "yes"), ("temperature", True),
                             ("top_p", "0.9"), ("max_tokens", 1.5),
                             ("max_tokens", True), ("n", 1.5)):
            with self.assertRaises(HTTPError, msg=field) as caught:
                self.request("/v1/chat/completions", dict(base, **{field: value}))
            self.assertEqual(caught.exception.code, 400)

        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/messages", {
                "model": "glm-test", "messages": base["messages"],
                "max_tokens": 1.5,
            })
        self.assertEqual(caught.exception.code, 400)

    def test_anthropic_error_shape_token_count_and_static_safety(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/messages", {
                "model": "glm-test", "messages": [{"role": "user", "content": "hi"}],
            })
        error = json.load(caught.exception)
        self.assertEqual(error["type"], "error")
        self.assertTrue(error["request_id"].startswith("req_"))
        self.assertIsNotNone(caught.exception.headers["request-id"])
        with self.request("/v1/messages/count_tokens", {
            "model": "glm-test", "messages": [{"role": "user", "content": "hello"}],
        }) as response:
            count = json.load(response)
            self.assertEqual(response.headers["x-colibri-token-count"], "estimated")
        self.assertGreater(count["input_tokens"], 0)
        with self.request("/") as response:
            self.assertEqual(response.read(), b"native dashboard")
        with self.assertRaises(HTTPError) as caught:
            self.request("/%2e%2e/" + self.outside.name + "/secret.txt")
        self.assertEqual(caught.exception.code, 404)

    def test_tool_calls_and_schema_aware_argument_types(self):
        tool = {"type": "function", "function": {
            "name": "lookup", "description": "lookup",
            "parameters": {"type": "object", "properties": {
                "q": {"type": "string"},
            }},
        }}
        body = {"model": "glm-test", "messages": [{"role": "user", "content": "use tool"}],
                "tools": [tool], "stream": False}
        with self.request("/v1/chat/completions", body) as response:
            result = json.load(response)
        call = result["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "lookup")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"q": "bird"})
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")

        response_tool = {"type": "function", "name": "lookup",
                         "parameters": tool["function"]["parameters"]}
        with self.request("/v1/responses", {
            "model": "glm-test", "input": "use tool", "tools": [response_tool],
        }) as response:
            result = json.load(response)
        self.assertEqual(result["output"][0]["type"], "function_call")
        self.assertEqual(json.loads(result["output"][0]["arguments"]), {"q": "bird"})
        first = result
        with self.request("/v1/responses", {
            "model": "glm-test", "previous_response_id": first["id"],
            "input": [{"type": "function_call_output",
                       "call_id": first["output"][0]["call_id"], "output": "sparrow"}],
            "reasoning": {"effort": "none"},
        }) as response:
            follow_up = json.load(response)
        self.assertEqual(follow_up["output"][0]["type"], "message")
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/responses", {
                "model": "glm-test", "previous_response_id": "resp_missing", "input": "x",
            })
        self.assertEqual(caught.exception.code, 404)

        anthropic_tool = {"name": "lookup", "description": "lookup",
                          "input_schema": tool["function"]["parameters"]}
        with self.request("/v1/messages", {
            "model": "glm-test", "messages": [{"role": "user", "content": "use tool"}],
            "max_tokens": 8, "tools": [anthropic_tool],
        }) as response:
            result = json.load(response)
        self.assertEqual(result["content"][0]["type"], "tool_use")
        self.assertEqual(result["content"][0]["input"], {"q": "bird"})
        self.assertEqual(result["stop_reason"], "tool_use")

        with self.request("/v1/messages", {
            "model": "glm-test",
            "messages": [{"role": "user", "content": "native tool syntax"}],
            "max_tokens": 8, "tools": [anthropic_tool],
        }) as response:
            native = json.load(response)
        self.assertEqual(native["content"][0]["text"], "I'll look that up.")
        self.assertEqual(native["content"][1]["type"], "tool_use")
        self.assertEqual(native["content"][1]["name"], "lookup")
        self.assertEqual(native["content"][1]["input"], {"q": "finch"})
        self.assertEqual(native["stop_reason"], "tool_use")

        with self.request("/v1/messages", {
            "model": "glm-test",
            "messages": [{"role": "user", "content": "native tool syntax"}],
            "max_tokens": 8, "tools": [anthropic_tool], "stream": True,
        }) as response:
            native_stream = response.read().decode()
        self.assertIn('"type":"tool_use"', native_stream)
        self.assertIn('"name":"lookup"', native_stream)
        self.assertIn('"partial_json":"{\\"q\\":\\"finch\\"}"', native_stream)
        self.assertIn('"stop_reason":"tool_use"', native_stream)

        with self.request("/api/chat", {**body, "stream": False}) as response:
            result = json.load(response)
        self.assertEqual(result["message"]["tool_calls"][0]["function"]["arguments"],
                         {"q": "bird"})

    def test_anthropic_deferred_tools_are_not_rendered_into_initial_prompt(self):
        tools = [
            {"name": "always_available", "description": "small",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "large_deferred", "description": "DEFERRED_SENTINEL",
             "input_schema": {"type": "object", "properties": {
                 "payload": {"type": "string", "description": "DEFERRED_SENTINEL"},
             }}, "defer_loading": True},
        ]
        with self.request("/v1/messages?beta=true", {
            "model": "glm-test", "messages": [{"role": "user",
                                                   "content": "check defer"}],
            "max_tokens": 4, "tools": tools,
        }) as response:
            result = json.load(response)
        self.assertEqual(result["content"][0]["text"], "deferred-ok")

        with self.request("/v1/messages?beta=true", {
            "model": "glm-test", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "check reference"},
                {"type": "tool_result", "tool_use_id": "search_1", "content": [
                    {"type": "tool_reference", "tool_name": "large_deferred"},
                ]},
            ]}], "max_tokens": 4, "tools": tools,
        }) as response:
            result = json.load(response)
        self.assertEqual(result["content"][0]["text"], "reference-expanded")

    def test_openai_stream_is_live_not_buffered(self):
        started = time.monotonic()
        with self.request("/v1/chat/completions", {
            "model": "glm-test", "messages": [{"role": "user", "content": "slow"}],
            "stream": True,
        }) as response:
            seen_first = None
            stream = ""
            while True:
                line = response.readline().decode()
                if not line:
                    break
                stream += line
                if '"content":"first"' in line and seen_first is None:
                    seen_first = time.monotonic() - started
        elapsed = time.monotonic() - started
        self.assertIsNotNone(seen_first)
        self.assertLess(seen_first, .3)
        self.assertGreater(elapsed, .45)
        self.assertIn('"content":"second"', stream)

    def test_zz_engine_exit_keeps_server_alive(self):
        body = {"model": "glm-test",
                "messages": [{"role": "user", "content": "exit-engine"}]}
        with self.assertRaises(HTTPError) as first:
            self.request("/v1/chat/completions", body)
        self.assertEqual(first.exception.code, 500)
        with self.assertRaises(HTTPError) as second:
            self.request("/v1/chat/completions", body)
        self.assertEqual(second.exception.code, 500)
        with self.request("/health") as response:
            self.assertEqual(json.load(response)["status"], "ok")
        self.assertIsNone(self.process.poll())

    def test_anthropic_stream_starts_before_tool_generation_finishes(self):
        started = time.monotonic()
        tool = {"name": "lookup", "input_schema": {"type": "object",
                                                       "properties": {}}}
        with self.request("/v1/messages?beta=true", {
            "model": "glm-test", "messages": [{"role": "user",
                                                   "content": "slow"}],
            "max_tokens": 4, "tools": [tool], "stream": True,
        }) as response:
            first_event = response.readline().decode()
            first_data = response.readline().decode()
            first_latency = time.monotonic() - started
            remainder = response.read().decode()
        self.assertEqual(first_event, "event: message_start\n")
        self.assertIn('"type":"message_start"', first_data)
        self.assertLess(first_latency, .3)
        self.assertGreater(time.monotonic() - started, .45)
        self.assertIn("event: message_stop", remainder)

    def test_tool_streams_incrementally_across_fragmented_markers(self):
        anthropic_tool = {"name": "lookup", "input_schema": {
            "type": "object", "properties": {"q": {"type": "string"}}}}
        started = time.monotonic()
        with self.request("/v1/messages", {
            "model": "glm-test", "messages": [{"role": "user",
                "content": "fragmented tool"}], "max_tokens": 16,
            "thinking": {"type": "disabled"}, "tools": [anthropic_tool],
            "stream": True,
        }) as response:
            first = response.readline().decode()
            response.readline()
            response.readline()
            first_delta = response.readline().decode()
            latency = time.monotonic() - started
            rest = response.read().decode()
        stream = first + first_delta + rest
        self.assertLess(latency, .3)
        self.assertIn('"text":"Before "', stream)
        self.assertIn('"type":"tool_use"', stream)
        self.assertIn('"partial_json":"{\\"q\\":\\"bird\\"}"', stream)
        self.assertNotIn("<tool_", stream)
        self.assertIn('"text":" After"', stream)

        openai_tool = {"type": "function", "function": {"name": "lookup",
            "parameters": {"type": "object", "properties": {
                "q": {"type": "string"}}}}}
        with self.request("/v1/chat/completions", {
            "model": "glm-test", "messages": [{"role": "user",
                "content": "fragmented tool"}], "max_tokens": 16,
            "tools": [openai_tool], "stream": True,
            "reasoning_effort": "none",
        }) as response:
            stream = response.read().decode()
        self.assertIn('"content":"Before "', stream)
        self.assertIn('"tool_calls"', stream)
        self.assertIn('"arguments":"{\\"q\\":\\"bird\\"}"', stream)
        self.assertNotIn("<tool_", stream)

        started = time.monotonic()
        with self.request("/v1/responses", {
            "model": "glm-test", "input": "fragmented tool",
            "max_output_tokens": 16, "tools": [openai_tool],
            "stream": True,
        }) as response:
            first = response.readline().decode()
            response.readline()
            first_delta = response.readline().decode()
            latency = time.monotonic() - started
            rest = response.read().decode()
        stream = first + first_delta + rest
        self.assertLess(latency, .3)
        self.assertIn('"delta":"Before "', stream)
        self.assertIn('"type":"response.function_call_arguments.delta"',
                      stream)
        self.assertIn('"delta":"{\\"q\\":\\"bird\\"}"', stream)
        self.assertIn('"type":"response.output_item.added"', stream)
        self.assertIn('"arguments":""', stream)
        self.assertIn('"type":"response.function_call_arguments.done"', stream)
        self.assertIn('"arguments":"{\\"q\\":\\"bird\\"}"', stream)
        self.assertIn('"type":"response.output_item.done"', stream)
        self.assertNotIn("<tool_", stream)
        payloads = [json.loads(line[len("data: "):])
                    for line in stream.splitlines()
                    if line.startswith("data: {")]
        lifecycle = [event["type"] for event in payloads
                     if event["type"].startswith("response.output_item") or
                     event["type"].startswith(
                         "response.function_call_arguments")]
        self.assertEqual(lifecycle, [
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
        ])

        with self.request("/api/chat", {
            "model": "glm-test", "messages": [{"role": "user",
                "content": "fragmented tool"}], "tools": [openai_tool],
            "stream": True, "think": False,
        }) as response:
            events = [json.loads(line) for line in response if line.strip()]
        self.assertEqual(events[0]["message"]["content"], "Before ")
        calls = [e["message"]["tool_calls"] for e in events
                 if e.get("message", {}).get("tool_calls")]
        self.assertEqual(calls[0][0]["function"]["arguments"], {"q": "bird"})
        self.assertTrue(events[-1]["done"])

        required_tool = {"type": "function", "function": {"name": "lookup",
            "parameters": {"type": "object", "properties": {
                "q": {"type": "string"}}, "required": ["q"]}}}
        with self.request("/v1/chat/completions", {
            "model": "glm-test", "messages": [{"role": "user",
                "content": "malformed orphan tool value"}],
            "tools": [required_tool], "reasoning_effort": "none",
        }) as response:
            result = json.load(response)
        call = result["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]),
                         {"q": "bird"})

        with self.request("/v1/chat/completions", {
            "model": "glm-test", "messages": [{"role": "user",
                "content": "two malformed orphan tool values"}],
            "tools": [required_tool], "reasoning_effort": "none",
        }) as response:
            result = json.load(response)
        calls = result["choices"][0]["message"]["tool_calls"]
        self.assertEqual([json.loads(call["function"]["arguments"])
                          for call in calls], [{"q": "bird"}, {"q": "tern"}])

    def test_multiple_streamed_tool_calls_have_unique_stable_ids(self):
        parameters = {"type": "object", "properties": {
            "q": {"type": "string"}}}
        openai_tool = {"type": "function", "function": {
            "name": "lookup", "parameters": parameters}}
        anthropic_tool = {"name": "lookup", "input_schema": parameters}

        with self.request("/v1/messages", {
            "model": "glm-test", "messages": [{"role": "user",
                "content": "multiple tools"}], "max_tokens": 16,
            "thinking": {"type": "disabled"}, "tools": [anthropic_tool],
            "stream": True,
        }) as response:
            events = [json.loads(line[6:]) for line in response
                      if line.startswith(b"data: ")]
        anthropic_ids = [
            event["content_block"]["id"] for event in events
            if event.get("type") == "content_block_start"
            and event["content_block"]["type"] == "tool_use"
        ]
        self.assertEqual(len(anthropic_ids), 2)
        self.assertEqual(len(set(anthropic_ids)), 2)

        with self.request("/v1/chat/completions", {
            "model": "glm-test", "messages": [{"role": "user",
                "content": "multiple tools"}], "max_tokens": 16,
            "tools": [openai_tool], "stream": True,
            "reasoning_effort": "none",
        }) as response:
            events = [json.loads(line[6:]) for line in response
                      if line.startswith(b"data: ") and b"[DONE]" not in line]
        deltas = [
            call for event in events for choice in event["choices"]
            for call in choice["delta"].get("tool_calls", [])
        ]
        self.assertEqual([call["index"] for call in deltas], [0, 1])
        self.assertEqual(len({call["id"] for call in deltas}), 2)

        with self.request("/v1/responses", {
            "model": "glm-test", "input": "multiple tools",
            "max_output_tokens": 16, "tools": [openai_tool],
            "stream": True,
        }) as response:
            events = [json.loads(line[6:]) for line in response
                      if line.startswith(b"data: ") and b"[DONE]" not in line]
        delta_ids = [
            event["item_id"] for event in events
            if event["type"] == "response.function_call_arguments.delta"
        ]
        completed = next(event["response"] for event in events
                         if event["type"] == "response.completed")
        completed_ids = [item["id"] for item in completed["output"]]
        self.assertEqual(len(set(delta_ids)), 2)
        self.assertEqual(delta_ids, completed_ids)
        self.assertEqual(len({item["call_id"]
                              for item in completed["output"]}), 2)

    def test_anthropic_stream_retains_split_utf8_tail(self):
        with self.request("/v1/messages", {
            "model": "glm-test", "messages": [{"role": "user",
                "content": "split utf8"}], "max_tokens": 4,
            "stream": True,
        }) as response:
            stream = response.read().decode("utf-8")
        self.assertIn('"text":"é"', stream)
        self.assertNotIn("�", stream)


class NativePlanDoctorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name) / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({
            "num_hidden_layers": 2, "n_routed_experts": 2,
            "kv_lora_rank": 4, "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 3, "v_head_dim": 5,
            "num_attention_heads": 2,
        }))
        (self.model / "tokenizer.json").write_text("{}")
        write_shard(self.model / "model.safetensors", [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.self_attn.q_a_proj.weight", 200),
            ("model.layers.1.mlp.experts.0.gate_proj.weight", 30),
            ("model.layers.1.mlp.experts.0.up_proj.weight", 30),
            ("model.layers.1.mlp.experts.1.gate_proj.weight", 30),
            ("model.layers.1.mlp.experts.1.up_proj.weight", 30),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args, check=True):
        return subprocess.run([str(ROOT / "coli-native"), *args],
                              universal_newlines=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=check)

    def test_help_does_not_require_an_engine(self):
        env = dict(os.environ, COLI_ENGINE="/does/not/exist")
        run = subprocess.run([str(ROOT / "coli-native"), "--help"], env=env,
                             universal_newlines=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=True)
        self.assertIn("coli serve", run.stdout)

    def test_native_plan_analyzes_model_and_emits_versioned_json(self):
        run = self.run_cli("plan", "--model", str(self.model), "--gpu", "none",
                           "--ram", "16", "--ctx", "32", "--json")
        plan = json.loads(run.stdout)
        self.assertEqual(plan["version"], 2)
        self.assertEqual(plan["model"]["dense_bytes"], 300)
        self.assertEqual(plan["model"]["expert_bytes"], 120)
        self.assertEqual(plan["model"]["expert_count"], 2)
        self.assertEqual(plan["model"]["per_cap_bytes"], 60)
        self.assertEqual(plan["tiers"]["ram"]["budget_bytes"], 16_000_000_000)
        self.assertEqual(plan["tiers"]["vram"]["devices"], [])

    def test_native_doctor_is_read_only_and_machine_readable(self):
        run = self.run_cli("doctor", "--model", str(self.model), "--gpu", "none",
                           "--ram", "16", "--ctx", "32", "--json", check=False)
        self.assertIn(run.returncode, (0, 1))
        report = json.loads(run.stdout)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(Path(report["model"]), self.model)
        self.assertIn(report["status"], ("ok", "warning", "error"))
        checks = {item["id"]: item for item in report["checks"]}
        self.assertEqual(checks["model.shards"]["status"], "pass")
        self.assertIsNotNone(report["plan"])

    def test_native_chat_uses_persistent_engine_protocol(self):
        env = dict(os.environ, COLI_ENGINE=str(ROOT / "tests/fake_mux_engine"))
        run = subprocess.run([
            str(ROOT / "coli-native"), "chat", "--model", str(self.model), "--ngen", "4",
        ], input="hello\n:reset\n:q\n", env=env, universal_newlines=True,
           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        self.assertIn("chat:hello", run.stdout)
        self.assertIn("memory cleared", run.stderr)


if __name__ == "__main__":
    unittest.main()
