#!/usr/bin/env python3
"""Format shim between Telegram voice notes and the NeMo-Speech.cpp HTTP server.

`nemo-speech serve` answers `POST /v1/audio/transcriptions` with an OpenAI-shaped
`{"text": "..."}`, which is exactly the contract whisper.cpp answers today — but it
only accepts *uncompressed RIFF/WAVE* (docs/api.md: "multipart form with the
uncompressed RIFF/WAVE file in `file`"). Telegram sends Opus in Ogg, and both of
this cluster's callers upload the attachment verbatim, so something has to decode
it. whisper.cpp did that in-process with `--convert`; here it is this file.

Position in the pod: this listens on the published port and the engine listens on
127.0.0.1 only, so nothing can reach the engine without coming through here.

Behaviour, in one paragraph: read the multipart body; pass a compliant WAV through
byte-for-byte and hand anything else to ffmpeg (`-ac 1 -ar 16000 pcm_s16le`, the
same three flags whisper.cpp's convert_to_wav uses); rebuild the form with only the
fields the engine documents; proxy the answer back with its status and content type
untouched. Failures answer in the engine's own error shape, `{"error": {...}}`,
with a 4xx — never 200 — so the callers' `curl -f` turns them into a failed entry
rather than into a transcript.

No third-party imports: the image is ubuntu:24.04 + python3 + ffmpeg and nothing
else, which keeps the build reproducible and the audit surface this size on purpose.
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Optional, Tuple

# Form fields the engine documents (docs/api.md, POST /v1/audio/transcriptions).
# Anything else in the upload is dropped: whisper.cpp's own knobs (`temperature`,
# `no_context`, `no_timestamps`, `initial_prompt`, `translate`, ...) are not
# understood here, and forwarding an unknown field is a coin flip about whether the
# engine ignores it or answers 400.
ENGINE_FIELDS = frozenset(
    {
        "model",
        "file",
        "language",
        "response_format",
        "automatic_punctuation",
        "verbatim",
        "profanity_filter",
        "speech_contexts",
        "prompt",
        "diarization",
        "max_speaker_count",
    }
)

# Endpoints forwarded verbatim (GET, no body). /v1/audio/transcriptions is handled
# separately; / is the playground, which --no-ui removes from the engine anyway and
# which stays unreachable here.
PASSTHROUGH_GET = ("/health", "/ready", "/version", "/v1/models")

WAV_MAGIC = re.compile(rb"^RIFF(.{4})WAVE$", re.DOTALL)

# Boundary for the form rebuilt on the way out. Fixed rather than random: it never
# leaves the pod's loopback, and a constant keeps the forwarded request diffable in
# engine-side logs.
BOUNDARY = b"parakeet-adapter-boundary"


class BadRequest(Exception):
    """A client error worth a 4xx; the message goes out in the engine's error shape."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------- #
# multipart parsing / rebuilding
# --------------------------------------------------------------------------- #


PARAMETER = re.compile(r'([^\s;=]+)\s*=\s*("(?:[^"\\]|\\.)*"|[^;]*)')


def header_parameters(raw: str) -> Tuple[str, dict]:
    """Split a header value into its value and its `; key="value"` parameters.

    `email.message.Message.get_params` is the obvious call and is the wrong one: it
    reads the *Content-Type* header no matter which header the message object was
    handed, so pointed at Content-Disposition it answers "no parameters" and every
    upload looks like it has no file in it.
    """
    value, _, rest = raw.partition(";")
    params: dict[str, str] = {}
    for match in PARAMETER.finditer(rest):
        name = match.group(1).strip().lower()
        body = match.group(2).strip()
        if len(body) >= 2 and body[0] == '"' and body[-1] == '"':
            body = body[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        params[name] = body
    return value.strip(), params


def boundary_of(content_type: Optional[str]) -> bytes:
    """The multipart boundary from a Content-Type header, as bytes."""
    if not content_type:
        raise BadRequest("expected a multipart/form-data upload")
    maintype, _, _ = content_type.partition("/")
    if maintype.strip().lower() != "multipart":
        raise BadRequest(f"expected multipart/form-data, got {content_type.split(';')[0]}")
    _, params = header_parameters(content_type)
    raw = params.get("boundary")
    if not raw:
        raise BadRequest("multipart upload declared no boundary")
    return raw.encode("ascii", "surrogateescape")


def parse_multipart(body: bytes, boundary: bytes) -> list[Tuple[dict, bytes]]:
    """Split a multipart body into (headers, payload) pairs.

    Hand-rolled rather than handed to the `email` package because that package is
    written for text mail and its payload handling of arbitrary binary (line-ending
    normalisation, transfer-encoding guesses) is exactly what a WAV cannot survive.
    RFC 2046 is unambiguous here: parts are separated by CRLF + "--" + boundary, and
    the delimiter belongs to the *following* part, so the payload ends where the
    CRLF before the next delimiter starts.
    """
    delimiter = b"\r\n--" + boundary
    # The body proper starts with "--<boundary>", i.e. without a leading CRLF, so
    # prepend one and every delimiter becomes the same byte string.
    buf = body if body.startswith(b"\r\n") else b"\r\n" + body

    # Occurrences of the delimiter, and only real ones: RFC 2046 requires CRLF or the
    # closing "--" to follow it, so bytes inside a WAV that merely start with the
    # boundary are content, not structure. `split` would cut there.
    starts: list[int] = []
    cursor = 0
    while True:
        found = buf.find(delimiter, cursor)
        if found < 0:
            break
        if buf[found + len(delimiter) : found + len(delimiter) + 2] in (b"\r\n", b"--"):
            starts.append(found)
            cursor = found + len(delimiter)
        else:
            cursor = found + 2

    parts: list[Tuple[dict, bytes]] = []
    for index, start in enumerate(starts):
        begins = start + len(delimiter)
        ends = starts[index + 1] if index + 1 < len(starts) else len(buf)
        chunk = buf[begins:ends]
        if chunk.startswith(b"--"):  # the closing "--\r\n"
            break
        raw_headers, sep, payload = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        headers: dict[str, str] = {}
        for line in raw_headers.split(b"\r\n"):
            name, colon, value = line.partition(b":")
            if colon:
                headers[name.decode("ascii", "replace").strip().lower()] = value.decode(
                    "ascii", "replace"
                ).strip()
        parts.append((headers, payload))
    return parts


def disposition(headers: dict) -> Tuple[Optional[str], Optional[str]]:
    """(field name, filename) from a part's Content-Disposition."""
    raw = headers.get("content-disposition")
    if not raw:
        return None, None
    _, params = header_parameters(raw)
    return params.get("name"), params.get("filename")


def build_multipart(fields: list[Tuple[str, str]], audio: Tuple[str, bytes], boundary: bytes) -> bytes:
    """Rebuild the form: text fields first, then the (possibly transcoded) file."""
    out = bytearray()
    for name, value in fields:
        out += b"--" + boundary + b"\r\n"
        out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += value.encode() + b"\r\n"
    filename, blob = audio
    out += b"--" + boundary + b"\r\n"
    out += (
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    out += blob + b"\r\n"
    out += b"--" + boundary + b"--\r\n"
    return bytes(out)


# --------------------------------------------------------------------------- #
# WAV inspection + ffmpeg
# --------------------------------------------------------------------------- #


def wav_format(blob: bytes) -> Optional[dict]:
    """The fmt fields that decide what to do with a WAV, or None if it is not one."""
    if not WAV_MAGIC.match(blob[:12]):
        return None
    pos = 12
    while pos + 8 <= len(blob):
        tag = blob[pos : pos + 4]
        size = int.from_bytes(blob[pos + 4 : pos + 8], "little")
        if tag == b"fmt " and pos + 8 + 16 <= len(blob):
            body = blob[pos + 8 : pos + 24]
            info = {
                "audio_format": int.from_bytes(body[0:2], "little"),
                "channels": int.from_bytes(body[2:4], "little"),
                "sample_rate": int.from_bytes(body[4:8], "little"),
                "bits": int.from_bytes(body[14:16], "little"),
                "bytes_per_second": int.from_bytes(body[8:12], "little"),
                "data_bytes": 0,
            }
            # The data chunk is normally next; walk anyway, some encoders pad.
            scan = pos + 8 + size + (size % 2)
            while scan + 8 <= len(blob):
                if blob[scan : scan + 4] == b"data":
                    info["data_bytes"] = int.from_bytes(blob[scan + 4 : scan + 8], "little")
                    break
                step = int.from_bytes(blob[scan + 4 : scan + 8], "little")
                scan += 8 + step + (step % 2)
            return info
        if size < 0:
            return None
        pos += 8 + size + (size % 2)  # chunks are word-aligned
    return None


def wav_duration(blob: bytes) -> Optional[float]:
    """Seconds of audio in a WAV, from its header — no decoding, no temp file."""
    info = wav_format(blob)
    if not info:
        return None
    rate = info["bytes_per_second"] or (
        info["sample_rate"] * info["channels"] * max(info["bits"], 1) // 8
    )
    if not rate:
        return None
    # A 4 GiB WAV wraps its size fields at 2^32-1; then fall back on the blob.
    data = min(info["data_bytes"], max(len(blob) - 44, 0)) or max(len(blob) - 44, 0)
    return data / rate


def probe_duration(blob: bytes, workdir: str, timeout: float) -> Optional[float]:
    """Seconds of audio in an arbitrary container, via ffprobe. None if unknown."""
    src = os.path.join(workdir, f"probe-{uuid.uuid4().hex}")
    try:
        with open(src, "wb") as handle:
            handle.write(blob)
        try:
            done = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", src,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None
        text = (done.stdout or b"").decode().strip()
        try:
            return float(text)
        except ValueError:
            return None
    finally:
        try:
            os.unlink(src)
        except OSError:
            pass


def is_engine_ready_wav(blob: bytes) -> bool:
    """True when the engine can decode this upload as it stands.

    Accepts what docs/api.md accepts: PCM16 or float32, 8-96 kHz, mono or stereo.
    Everything else — Ogg/Opus from Telegram, MP3/M4A from a file upload, an
    8-bit or multichannel WAV — goes through ffmpeg.
    """
    info = wav_format(blob)
    if not info:
        return False
    return (
        info["audio_format"] in (1, 3)  # 1 = integer PCM, 3 = IEEE float
        and info["bits"] in (16, 32)
        and 1 <= info["channels"] <= 2
        and 8000 <= info["sample_rate"] <= 96000
    )


def transcode_to_wav(blob: bytes, workdir: str, timeout: float) -> bytes:
    """Decode any container ffmpeg knows into mono 16 kHz PCM16 WAV."""
    src = os.path.join(workdir, f"in-{uuid.uuid4().hex}")
    dst = os.path.join(workdir, f"out-{uuid.uuid4().hex}.wav")
    try:
        with open(src, "wb") as handle:
            handle.write(blob)
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            src,
            # The voice note is the first audio stream; -vn would be the mirror
            # image of that and is unnecessary. Matched to whisper.cpp's --convert
            # (-ar 16000 -ac 1 pcm_s16le), which is the model's own input geometry.
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            dst,
        ]
        started = time.monotonic()
        try:
            done = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise BadRequest(f"ffmpeg exceeded {timeout:.0f}s", 504) from exc
        if done.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
            detail = (done.stderr or b"").decode("utf-8", "replace").strip()[-400:]
            raise BadRequest(f"ffmpeg could not decode the upload: {detail or 'no audio stream'}")
        with open(dst, "rb") as handle:
            out = handle.read()
        log(
            f"transcoded {len(blob)} bytes -> {len(out)} bytes wav in "
            f"{(time.monotonic() - started) * 1000:.0f} ms"
        )
        return out
    finally:
        for path in (src, dst):
            try:
                os.unlink(path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# the server
# --------------------------------------------------------------------------- #


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "parakeet-adapter/0.2.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------- #
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib name
        log(f"{self.client_address[0]} {fmt % args}")

    def _reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        payload = json.dumps({"error": {"message": message, "type": "invalid_request_error"}})
        self._reply(status, payload.encode(), "application/json")

    def _proxy(self, method: str, path: str, body: Optional[bytes], content_type: Optional[str]) -> None:
        """Forward one request to the engine and relay its answer untouched."""
        cfg = self.server.cfg  # type: ignore[attr-defined]
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type
        try:
            connection = http.client.HTTPConnection(
                cfg.upstream_host, cfg.upstream_port, timeout=cfg.upstream_timeout
            )
            try:
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                payload = response.read()
                status = response.status
                ctype = response.getheader("Content-Type", "application/json")
            finally:
                connection.close()
        except (OSError, http.client.HTTPException) as exc:
            # The engine not answering is a server-side failure: 502, so `curl -f`
            # exits 22 and no transcript is invented from an empty body.
            self._error(502, f"upstream {cfg.upstream_host}:{cfg.upstream_port} unavailable: {exc}")
            return
        self._reply(status, payload, ctype)

    # -- routes ------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        path = self.path.split("?", 1)[0]
        if path in PASSTHROUGH_GET:
            self._proxy("GET", path, None, None)
        elif path == "/healthz":
            # Adapter-local liveness: answers even while the engine is loading, so
            # a slow first model load can be told apart from a dead sidecar.
            self._reply(200, b'{"status":"ok","component":"adapter"}', "application/json")
        else:
            self._error(404, f"no route {path}")

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        cfg = self.server.cfg  # type: ignore[attr-defined]
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/audio/transcriptions", "/v1/audio/translations"):
            self._error(404, f"no route {path}")
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._error(400, "upload carried no Content-Length")
            return
        if length > cfg.max_upload_bytes:
            # Refuse the upload, but drain a little first: a client that is already
            # mid-write on a 2 MiB body otherwise gets a reset instead of the 413,
            # and the bytes left in the socket get parsed as the next request line.
            remaining = min(length, 2 * 1024 * 1024)
            while remaining > 0:
                chunk = self.rfile.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True
            self._error(
                413,
                f"upload is {length} bytes, over the {cfg.max_upload_bytes} byte cap",
            )
            return

        started = time.monotonic()
        try:
            body = self.rfile.read(length)
            boundary = boundary_of(self.headers.get("Content-Type"))
            parts = parse_multipart(body, boundary)
            fields: list[Tuple[str, str]] = []
            audio: Optional[Tuple[str, bytes]] = None
            for headers, payload in parts:
                name, filename = disposition(headers)
                if name is None:
                    continue
                if name == "file" or (audio is None and filename is not None):
                    audio = (filename or "upload", payload)
                    continue
                lowered = name.lower()
                if lowered in ENGINE_FIELDS and lowered != "file":
                    fields.append((lowered, payload.decode("utf-8", "replace")))
                else:
                    log(f"dropped unsupported form field {name!r}")
            if audio is None:
                raise BadRequest("upload carried no file field")
            name, blob = audio
            if not blob:
                raise BadRequest("uploaded file was empty")

            # Duration before size: 32 MiB of Opus is ~110 minutes of speech, so the
            # byte cap says nothing about what the encoder is about to be asked to
            # attend over. Parakeet TDT is full-utterance — attention over the whole
            # recording, memory quadratic in its length — and an over-long note is
            # OOMKilled, not refused: measured at 280 s and 420 s against
            # limits.memory 3Gi, engine exit 137. See README.md -> Long recordings.
            ready_wav = is_engine_ready_wav(blob)
            duration = (
                wav_duration(blob)
                if ready_wav
                else probe_duration(blob, cfg.workdir, cfg.ffmpeg_timeout)
            )
            if duration is not None and duration > cfg.max_audio_seconds:
                raise BadRequest(
                    f"recording is {duration:.0f}s, over the {cfg.max_audio_seconds:.0f}s "
                    f"this service will attempt (see README.md -> Long recordings)",
                    413,
                )

            if ready_wav:
                prepared = blob
            else:
                prepared = transcode_to_wav(blob, cfg.workdir, cfg.ffmpeg_timeout)
            # The authoritative check, on what the engine is actually about to decode:
            # ffprobe reports "N/A" for some streams, and a header size field can lie.
            decoded = wav_duration(prepared)
            if decoded is not None and decoded > cfg.max_audio_seconds:
                raise BadRequest(
                    f"decoded audio is {decoded:.0f}s, over the "
                    f"{cfg.max_audio_seconds:.0f}s this service will attempt", 413
                )
        except BadRequest as exc:
            self._error(exc.status, exc.message)
            return
        except Exception as exc:  # noqa: BLE001 - a sidecar must never die on one request
            self._error(500, f"{type(exc).__name__}: {exc}")
            return

        forwarded = build_multipart(fields, (name, prepared), BOUNDARY)
        log(
            f"POST {path} {len(blob)} -> {len(prepared)} bytes, fields="
            f"{sorted(f[0] for f in fields)}"
        )
        self._proxy(
            "POST", path, forwarded, f"multipart/form-data; boundary={BOUNDARY.decode()}"
        )
        log(f"POST {path} finished in {(time.monotonic() - started) * 1000:.0f} ms")


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, cfg):
        super().__init__(address, handler)
        self.cfg = cfg


def log(message: str) -> None:
    print(f"[adapter] {message}", flush=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--listen", default="0.0.0.0:8080", help="host:port to publish")
    parser.add_argument("--upstream", default="127.0.0.1:8081", help="nemo-speech host:port")
    parser.add_argument(
        "--max-upload-mb", type=int, default=32, help="request body cap, in MiB"
    )
    parser.add_argument(
        "--ffmpeg-timeout", type=float, default=60.0, help="per-upload decode deadline"
    )
    parser.add_argument(
        "--upstream-timeout", type=float, default=180.0, help="engine response deadline"
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=180.0,
        help="longest recording accepted, measured in audio seconds",
    )
    parser.add_argument("--workdir", default="/tmp", help="scratch for ffmpeg")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    listen_host, _, listen_port = args.listen.rpartition(":")
    upstream_host, _, upstream_port = args.upstream.rpartition(":")
    args.listen_host = listen_host or "0.0.0.0"
    args.listen_port = int(listen_port)
    args.upstream_host = upstream_host or "127.0.0.1"
    args.upstream_port = int(upstream_port)
    args.max_upload_bytes = args.max_upload_mb * 1024 * 1024

    os.makedirs(args.workdir, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        log("ffmpeg is not on PATH — non-WAV uploads will be refused with 400")

    http.server.ThreadingHTTPServer.allow_reuse_address = True
    server = Server((args.listen_host, args.listen_port), Handler, args)

    stopping = threading.Event()

    def shutdown(signum, _frame):
        if stopping.is_set():
            return
        stopping.set()
        log(f"received signal {signum}, draining")
        threading.Thread(target=server.shutdown, daemon=True).start()

    for signal_number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signal_number, shutdown)

    log(
        f"listening on {args.listen_host}:{args.listen_port}, upstream "
        f"{args.upstream_host}:{args.upstream_port}, cap {args.max_upload_mb} MiB"
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
