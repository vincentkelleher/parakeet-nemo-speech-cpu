#!/usr/bin/env python3
"""Tests for adapter.py — run them anywhere, no cluster and no model required.

    python3 test_adapter.py                       # needs ffmpeg for 2 of them

    # and again inside the image, which is the one that matters — it is what caught a
    # bundled libstdc++ shadowing the system one and killing ffmpeg (README.md):
    docker buildx build --platform=linux/amd64 -t parakeet:ci .
    docker run --rm --platform linux/amd64 -v "$PWD:/t:ro" --entrypoint python3 \
        parakeet:ci -m unittest discover -s /t -p "test_*.py"

The adapter has two jobs that can be checked without ever loading 681 MiB of weights:
split a multipart upload correctly (a WAV must come out byte-identical, or a
transcription fails for a reason no log names) and refuse what it cannot decode with
a 4xx (or a failed note becomes a transcript). Both need a peer to talk to, so this
starts a stub upstream that answers with what it received, and points the adapter at
it. Requires ffmpeg on PATH for the two cases that exercise transcoding; those skip
themselves when it is not.
"""

from __future__ import annotations

import http.client
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapter  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HERE = tempfile.mkdtemp(prefix="adapter-test-")


def make_wav(path: str, seconds: float = 1.0, rate: int = 16000, channels: int = 1) -> str:
    """A sine-wave WAV, the kind the engine accepts as it stands."""
    target = os.path.join(path, f"{uuid.uuid4().hex}.wav")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-ar", str(rate), "-ac", str(channels), "-c:a", "pcm_s16le", target,
        ],
        check=True,
    )
    return target


class StubUpstream(http.server.ThreadingHTTPServer):
    daemon_threads = True


def start_stub() -> tuple[StubUpstream, list[dict]]:
    """An upstream that records each request and replies like the engine would."""
    seen: list[dict] = []

    class Stub(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/ready":
                body = b'{"ready": true, "device": "cpu"}'
            else:
                body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            boundary = adapter.boundary_of(self.headers.get("Content-Type"))
            parts = adapter.parse_multipart(raw, boundary)
            fields = {}
            audio = None
            for headers, payload in parts:
                name, filename = adapter.disposition(headers)
                if name == "file":
                    audio = (filename, payload)
                else:
                    fields[name] = payload.decode()
            seen.append({"fields": fields, "audio": audio, "raw": raw})
            body = json.dumps({"text": " bonjour"}).encode()
            status = 200
            if fields.get("response_format") == "text":
                body, ctype = b" bonjour", "text/plain; charset=utf-8"
            else:
                ctype = "application/json"
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

    stub = StubUpstream(("127.0.0.1", 0), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    return stub, seen


class AdapterTest(unittest.TestCase):
    stub: StubUpstream
    server: adapter.Server
    port: int
    seen: list[dict]

    @classmethod
    def setUpClass(cls):
        cls.stub, cls.seen = start_stub()
        argv = [
            "--listen", "127.0.0.1:0",
            "--upstream", f"127.0.0.1:{cls.stub.server_address[1]}",
            "--max-upload-mb", "1",
            "--workdir", HERE,
        ]
        args = adapter.parse_args(argv)
        args.listen_host, args.listen_port = "127.0.0.1", 0
        args.upstream_host, args.upstream_port = "127.0.0.1", cls.stub.server_address[1]
        args.max_upload_bytes = args.max_upload_mb * 1024 * 1024
        cls.server = adapter.Server(("127.0.0.1", 0), adapter.Handler, args)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.stub.shutdown()
        cls.stub.server_close()

    def setUp(self):
        self.seen.clear()

    # -- helpers ----------------------------------------------------------- #
    def post(self, body: bytes, content_type: str, path="/v1/audio/transcriptions"):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        connection.request("POST", path, body=body, headers={"Content-Type": content_type})
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, response.getheader("Content-Type"), payload

    def get(self, path):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        connection.request("GET", path)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, payload

    def form(self, fields: dict, file_field="file", blob=None, filename="note.ogg"):
        boundary = "test-boundary"
        out = bytearray()
        for name, value in fields.items():
            out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        if blob is not None:
            out += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
                "\r\n"
            ).encode() + blob + b"\r\n"
        out += f"--{boundary}--\r\n".encode()
        return bytes(out), f"multipart/form-data; boundary={boundary}"

    # -- pass-through ------------------------------------------------------ #
    def test_compliant_wav_reaches_the_engine_byte_for_byte(self):
        source = Path(make_wav(HERE, seconds=0.5))
        blob = source.read_bytes()
        body, ctype = self.form({}, blob=blob, filename="clip.wav")
        status, _, payload = self.post(body, ctype)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["text"], " bonjour")
        sent_name, sent = self.seen[0]["audio"]
        self.assertEqual(sent_name, "clip.wav")
        self.assertEqual(sent, blob, "a WAV the engine can read must not be re-encoded")

    def test_text_response_format_is_forwarded_and_the_body_relayed(self):
        source = Path(make_wav(HERE, seconds=0.25))
        body, ctype = self.form(
            {"model": "whisper-1", "response_format": "text", "language": "fr"},
            blob=source.read_bytes(),
            filename="note.wav",
        )
        status, content_type, payload = self.post(body, ctype)
        self.assertEqual(status, 200)
        self.assertEqual(payload, b" bonjour")
        self.assertTrue(content_type.startswith("text/plain"), content_type)
        fields = self.seen[0]["fields"]
        self.assertEqual(fields["response_format"], "text")
        self.assertEqual(fields["language"], "fr")
        self.assertEqual(fields["model"], "whisper-1")

    # -- transcoding ------------------------------------------------------- #
    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
    def test_ogg_opus_becomes_mono_16k_pcm16_wav(self):
        ogg = os.path.join(HERE, f"{uuid.uuid4().hex}.ogg")
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-c:a", "libopus", "-b:a", "24k", "-ar", "48000", "-ac", "1", ogg,
            ],
            check=True,
        )
        body, ctype = self.form({}, blob=Path(ogg).read_bytes(), filename="voice.ogg")
        status, _, payload = self.post(body, ctype)
        self.assertEqual(status, 200, payload)
        _, forwarded = self.seen[0]["audio"]
        self.assertTrue(adapter.WAV_MAGIC.match(forwarded[:12]), "engine got a non-WAV")
        self.assertTrue(
            adapter.is_engine_ready_wav(forwarded),
            "forwarded WAV is outside what the engine accepts",
        )
        # 1 second of mono 16 kHz PCM16 = 44-byte header + 32 000 bytes of samples.
        # Tolerant by a whisker rather than exact: Opus decoders emit the encoder's
        # look-ahead samples too, so a 1 s clip decodes a few samples long.
        self.assertAlmostEqual(len(forwarded), 32_044, delta=200, msg=f"got {len(forwarded)}")

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
    def test_stereo_wav_passes_through_and_8bit_does_not(self):
        # docs/api.md accepts stereo and downmixes it, so re-encoding here would be
        # a wasted ffmpeg run on every request from a caller that sends stereo.
        stereo = Path(make_wav(HERE, seconds=0.5, rate=48000, channels=2))
        body, ctype = self.form({}, blob=stereo.read_bytes(), filename="stereo.wav")
        self.assertEqual(self.post(body, ctype)[0], 200)
        self.assertEqual(self.seen[0]["audio"][1], stereo.read_bytes())

        eight_bit = os.path.join(HERE, f"{uuid.uuid4().hex}.wav")
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(stereo), "-c:a", "pcm_u8", eight_bit,
            ],
            check=True,
        )
        body, ctype = self.form({}, blob=Path(eight_bit).read_bytes(), filename="u8.wav")
        self.assertEqual(self.post(body, ctype)[0], 200)
        _, forwarded = self.seen[0]["audio"]
        self.assertNotEqual(forwarded, Path(eight_bit).read_bytes())
        self.assertTrue(adapter.is_engine_ready_wav(forwarded), "engine got 8-bit audio")

    # -- errors ------------------------------------------------------------ #
    def test_undecodable_upload_is_400_not_200(self):
        body, ctype = self.form({}, blob=b"\x00\x01\x02\x03garbage", filename="note.ogg")
        status, _, payload = self.post(body, ctype)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(payload)["error"]["type"], "invalid_request_error")
        self.assertEqual(self.seen, [], "an undecodable upload must never reach the engine")

    def test_upload_without_a_file_is_400(self):
        body, ctype = self.form({"response_format": "text"}, blob=None)
        status, _, payload = self.post(body, ctype)
        self.assertEqual(status, 400)
        self.assertIn("no file", json.loads(payload)["error"]["message"])

    def test_oversize_upload_is_413(self):
        body, ctype = self.form({}, blob=b"x" * (2 * 1024 * 1024), filename="big.wav")
        status, _, payload = self.post(body, ctype)
        self.assertEqual(status, 413)
        self.assertIn("over the", json.loads(payload)["error"]["message"])

    def test_unknown_route_is_404(self):
        body, ctype = self.form({}, blob=b"RIFF\x00\x00\x00\x00WAVE", filename="x.wav")
        self.assertEqual(self.get("/v1/audio/nonsense")[0], 404)
        status, _, _ = self.post(body, ctype, path="/v1/load")
        self.assertEqual(status, 404, "the engine's model-loading route must stay unreachable")

    def test_whisper_only_fields_are_dropped_not_forwarded(self):
        source = Path(make_wav(HERE, seconds=0.2))
        body, ctype = self.form(
            {"temperature": "0.0", "no_context": "true", "initial_prompt": "Alice, Bob"},
            blob=source.read_bytes(),
            filename="note.wav",
        )
        status, _, _ = self.post(body, ctype)
        self.assertEqual(status, 200)
        self.assertEqual(self.seen[0]["fields"], {}, "whisper-only knobs leaked to the engine")

    # -- proxies ----------------------------------------------------------- #
    def test_readiness_is_the_engines_readiness(self):
        status, payload = self.get("/ready")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["device"], "cpu")

    def test_healthz_answers_without_the_engine(self):
        status, payload = self.get("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["component"], "adapter")

    def test_dead_upstream_is_502_not_an_empty_200(self):
        """The callers rely on a non-2xx: `-f` is what keeps a failure out of the chat."""
        args = adapter.parse_args(["--workdir", HERE])
        args.listen_host, args.listen_port = "127.0.0.1", 0
        args.upstream_host, args.upstream_port = "127.0.0.1", 1  # nothing there
        args.max_upload_bytes = 1024 * 1024
        server = adapter.Server(("127.0.0.1", 0), adapter.Handler, args)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
            connection.request("GET", "/ready")
            response = connection.getresponse()
            payload = response.read()
            connection.close()
            self.assertEqual(response.status, 502)
            self.assertIn("unavailable", json.loads(payload)["error"]["message"])
        finally:
            server.shutdown()
            server.server_close()


    def test_quoted_boundary_and_filename_with_spaces(self):
        source = Path(make_wav(HERE, seconds=0.2))
        boundary = "----=_Part_1234_5678.9012"
        out = bytearray()
        out += (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="voice message.wav"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + source.read_bytes() + b"\r\n"
        out += f"--{boundary}--\r\n".encode()
        status, _, payload = self.post(bytes(out), f'multipart/form-data; boundary="{boundary}"')
        self.assertEqual(status, 200, payload)
        self.assertEqual(self.seen[0]["audio"][0], "voice message.wav")


class DurationCapTest(unittest.TestCase):
    """The guard that keeps a 6-minute recording from OOMKilling the engine."""

    @classmethod
    def setUpClass(cls):
        if not HAVE_FFMPEG:
            raise unittest.SkipTest("ffmpeg not installed")
        args = adapter.parse_args(["--workdir", HERE, "--max-audio-seconds", "2.5"])
        args.listen_host, args.listen_port = "127.0.0.1", 0
        args.upstream_host, args.upstream_port = "127.0.0.1", 1
        args.max_upload_bytes = 64 * 1024 * 1024
        cls.server = adapter.Server(("127.0.0.1", 0), adapter.Handler, args)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.wav = Path(make_wav(HERE, seconds=5.0)).read_bytes()
        ogg = os.path.join(HERE, "long.ogg")
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
                "-c:a", "libopus", "-ac", "1", ogg,
            ],
            check=True,
        )
        cls.ogg = Path(ogg).read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, blob, filename):
        boundary = "cap-boundary"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f'filename="{filename}"\r\n\r\n'
        ).encode() + blob + f"\r\n--{boundary}--\r\n".encode()
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=60)
        connection.request(
            "POST", "/v1/audio/transcriptions", body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, payload

    def test_header_gives_the_length_of_a_wav(self):
        self.assertAlmostEqual(adapter.wav_duration(self.wav), 5.0, delta=0.05)
        self.assertIsNone(adapter.wav_duration(b"OggS\x00\x02"))

    def test_long_wav_is_refused_from_its_header_alone(self):
        status, payload = self.post(self.wav, "lecture.wav")
        self.assertEqual(status, 413)
        self.assertIn("over the", json.loads(payload)["error"]["message"])

    def test_long_ogg_is_refused_before_it_is_decoded(self):
        status, payload = self.post(self.ogg, "note.ogg")
        self.assertEqual(status, 413, payload)
        self.assertIn("recording is 5s, over the 2s", json.loads(payload)["error"]["message"])

    def test_short_upload_still_reaches_the_upstream(self):
        short = Path(make_wav(HERE, seconds=1.0)).read_bytes()
        status, payload = self.post(short, "ping.wav")
        # 502 rather than 413: the upstream here is a dead port on purpose, so 502 is
        # what "the guard let it through" looks like.
        self.assertEqual(status, 502, payload)


class ParserTest(unittest.TestCase):
    """The multipart splitter on its own, because bytes inside audio are not MIME."""

    @staticmethod
    def assemble(pairs: list[tuple[str, bytes, dict]]) -> tuple[bytes, bytes]:
        boundary = b"gc2f7a1"
        body = bytearray()
        for name, payload, extra in pairs:
            body += b"--" + boundary + b"\r\n"
            disposition = f'form-data; name="{name}"'
            if "filename" in extra:
                disposition += f'; filename="{extra["filename"]}"'
            body += f"Content-Disposition: {disposition}\r\n".encode()
            if "content-type" in extra:
                body += f'Content-Type: {extra["content-type"]}\r\n'.encode()
            body += b"\r\n" + payload + b"\r\n"
        body += b"--" + boundary + b"--\r\n"
        return bytes(body), boundary

    def test_two_parts_round_trip(self):
        audio = b"RIFF\x10\x00\x00\x00WAVEfmt " + bytes(range(256))
        body, boundary = self.assemble(
            [
                ("model", b"whisper-1", {}),
                ("file", audio, {"filename": "n.wav", "content-type": "audio/wav"}),
            ]
        )
        parts = adapter.parse_multipart(body, boundary)
        self.assertEqual(len(parts), 2)
        self.assertEqual(adapter.disposition(parts[0][0]), ("model", None))
        self.assertEqual(parts[0][1], b"whisper-1")
        self.assertEqual(adapter.disposition(parts[1][0]), ("file", "n.wav"))
        self.assertEqual(parts[1][1], audio)

    def test_boundary_lookalike_inside_the_payload_is_content(self):
        """A payload that starts a line with the boundary must not truncate itself.

        RFC 2046 only makes a delimiter out of CRLF + "--" + boundary when CRLF or the
        closing "--" follows, and a random client boundary cannot collide with audio
        bytes in practice — but `bytes.split` does not know that, and a cut here would
        silently transcribe the first half of a recording.
        """
        payload = b"AAAA\r\n--gc2f7a1-prefix\r\nBBBB\r\n--gc2f7a1\x00\r\nCCCC"
        body, boundary = self.assemble([("file", payload, {"filename": "a.bin"})])
        parts = adapter.parse_multipart(body, boundary)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0][1], payload)

    def test_rebuilt_form_is_parsable_by_the_same_code(self):
        fields = [("response_format", "text"), ("language", "fr")]
        audio = b"RIFF\x04\x00\x00\x00WAVE"
        rebuilt = adapter.build_multipart(fields, ("voice message.wav", audio), b"bnd")
        parts = adapter.parse_multipart(rebuilt, b"bnd")
        self.assertEqual(len(parts), 3)
        self.assertEqual([adapter.disposition(h)[0] for h, _ in parts], ["response_format", "language", "file"])
        self.assertEqual(parts[2][1], audio)
        self.assertEqual(adapter.disposition(parts[2][0]), ("file", "voice message.wav"))

    def test_a_non_wav_is_not_mistaken_for_one(self):
        # Ogg/Opus, the shape Telegram delivers: magic "OggS".
        self.assertFalse(adapter.is_engine_ready_wav(b"OggS\x00\x02\x00\x00\x00\x00\x00\x00"))
        # RIFF but AVI, not WAVE.
        self.assertFalse(adapter.is_engine_ready_wav(b"RIFF\x00\x00\x00\x00AVI LIST"))
        self.assertFalse(adapter.is_engine_ready_wav(b"short"))


if __name__ == "__main__":
    if not HAVE_FFMPEG:
        print("note: ffmpeg not on PATH, transcoding tests will skip", file=sys.stderr)
    unittest.main(verbosity=2, exit=False)
    shutil.rmtree(HERE, ignore_errors=True)
