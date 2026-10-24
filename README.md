# Parakeet TDT on CPU

Turn voice notes into text on a machine with no GPU, using an OpenAI-compatible
`POST /v1/audio/transcriptions`. NVIDIA's
[NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) runtime and its published
CPU build of [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
do the recognising; one stdlib-only file, `adapter.py`, is what makes it usable.

```
curl -F "file=@note.ogg" localhost:8080/v1/audio/transcriptions
{"text":"Pendant le second siècle je fis serment d'ouvrir tous les trésors de la terre …"}
```

It exists because two things about the runtime are individually reasonable and jointly
annoying:

1. **`nemo-speech serve` accepts RIFF/WAVE uploads only.** Mono or stereo, PCM16 or
   float32, 8–96 kHz — and nothing else. Everything real arrives as Opus in Ogg
   (Telegram, Signal, WhatsApp, webm from a browser recorder), and callers that speak
   the OpenAI API upload the attachment verbatim. `adapter.py` decodes with ffmpeg and
   forwards a compliant WAV untouched, so it costs nothing on the happy path.
2. **Upstream's `docker/Dockerfile` is not for a CPU box.** Its builder stage is
   `nvcr.io/nvidia/cuda:<ver>-devel-ubuntu24.04` (~8 GB) whatever it compiles, because
   one Dockerfile serves every backend. Meanwhile NVIDIA publishes
   `nemo-speech-<ver>-linux-x86_64-cpu.tar.gz` — the same artifact CI builds and tests,
   with a `.sha256` beside it. This image installs that archive, verifies its checksum
   at build time, and adds `ffmpeg`, `python3` and `curl`: **590 MB** total.

The result runs at **RTF ≈ 0.15 on a 2018 Intel i7-8700T** (6 cores, no AVX-512) with
**4 % word errors on French** broadcast speech, in a container capped at 4 CPUs and
3 GiB. CPU inference here is not a compromise; it is the sensible choice for short
recordings, and the reason a GPU has never been required for this.

## Get the image

Published by GitHub Actions to this repository's own GHCR namespace, public, pulled
anonymously — no `docker login` anywhere in the quick start below:

| Tag | What it is |
| --- | --- |
| `:0.2.0` | the version of NeMo-Speech.cpp these numbers were measured on |
| `:latest` | moves with each version tag |
| `:edge` | last merge to `main`; may not match the README |

```bash
docker pull ghcr.io/vincentkelleher/parakeet-nemo-speech-cpu:0.2.0
```

Building from source is one command and produces the same thing, because the build
verifies the sha256 of the NVIDIA archive it downloads before unpacking it — so a green
build from either path means the same binaries, and a changed upstream asset fails the
build rather than the container:

```bash
docker buildx build --platform=linux/amd64 \
    -t ghcr.io/vincentkelleher/parakeet-nemo-speech-cpu:0.2.0 --load .
```

## Quick start

amd64 only — the runtime ships `linux-x86_64`. On Apple Silicon everything below runs
under emulation, which works and is roughly 2–3× slower than native.

```bash
# 1. the model, 681 MiB, into ./models (gitignored, never baked into an image)
mkdir -p models && curl -L -o models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.q8_0.gguf
echo 'e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e  models/parakeet-tdt-0.6b-v3.q8_0.gguf' | sha256sum -c -

# 2. run (the image pulls from GHCR; build it locally instead if you prefer)
docker compose up -d

# 3. wait for the weights (~10-30 s native, longer emulated), then transcribe
curl localhost:8080/ready       # {"capabilities":["asr"],"device":"cpu","ready":true}
curl -F "file=@note.ogg" localhost:8080/v1/audio/transcriptions
```

`docker compose up` deliberately starts **two containers that share a network
namespace**, because that is the shape of the problem: the recognizer has no
authentication whatsoever — an unauthenticated `POST /v1/audio/transcriptions` on an
open port transcribes whatever anyone hands it, plus a playground UI — so it binds
`127.0.0.1` and never gets a published port, and the adapter is the only thing the host
can reach. Running one container with both processes works identically if you would
rather not have two:

```bash
docker run --rm -p 8080:8080 -v "$PWD/models:/models:ro" --platform linux/amd64 \
  ghcr.io/vincentkelleher/parakeet-nemo-speech-cpu:0.2.0 bash -c '
    /opt/nemo-speech/bin/nemo-speech serve --host 127.0.0.1 --port 8081 \
      --asr-model /models/parakeet-tdt-0.6b-v3.q8_0.gguf --device cpu --no-ui &
    python3 /opt/adapter/adapter.py --listen 0.0.0.0:8080 --upstream 127.0.0.1:8081'
```

## The API

Everything below was run against this recipe, not read out of a spec.

| Route | Notes |
| --- | --- |
| `POST /v1/audio/transcriptions` | multipart, field `file`. The only interesting one. |
| `GET /v1/models` | OpenAI-shaped listing of the single loaded model. `model` in a request is accepted and ignored. |
| `GET /ready` | the engine's own: 503 until the weights are loaded, then `{"capabilities":["asr"],"device":"cpu","ready":true}`. Point readiness probes here. |
| `GET /health`, `GET /version` | engine's, forwarded |
| `GET /healthz` | the adapter's own answer, so a wedged engine restarts the engine and not the sidecar |

`response_format`: `json` (default, `{"text": "…"}`), `text` (bare transcript),
`verbose_json` (duration + segments + words), `srt`, `vtt`. Anything else is
`400 response_format must be json, verbose_json, text, srt, or vtt` — relayed from the
engine with its status and shape intact.

```bash
curl -F "file=@note.ogg" -F "response_format=text" localhost:8080/v1/audio/transcriptions
curl -F "file=@speech.wav" -F "response_format=srt" localhost:8080/v1/audio/transcriptions
```

Input containers accepted, all measured through this adapter: **Ogg/Opus** (Telegram
voice notes), **MP3**, **M4A/AAC**, **WAV** at 8/16/44.1 kHz, mono or stereo, PCM16 or
float32. Anything ffmpeg decodes works; ffmpeg's output is always mono 16 kHz PCM16,
which is what the encoder wants anyway.

**A failure is never a transcript.** Over-long uploads, undecodable files, missing
`file` parts and an unreachable engine come back `4xx`/`5xx` with a
`{"error": {"message", "type", "param", "code"}}` body — never `200` with an empty
`text`. That is a deliberate contract, not decoration: the common client is
`curl -f … -F response_format=text`, where a 200 with an empty body becomes a confident
empty answer and a 4xx becomes a non-zero exit. Checked:

```
curl: (22) The requested URL returned error: 413
```

## What the adapter actually does

- **Parses multipart properly.** `email.message.Message` is not a multipart parser and
  mishandles boundaries that appear inside another boundary's prefix, or quoted
  filenames — that class of bug produces a corrupt WAV and a transcription that fails
  for a reason no log names. There are tests for boundary lookalikes and quoting.
- **Passes a compliant WAV through byte-for-byte**, so a client that already sends WAV
  pays one copy and no re-encode.
- **Decodes everything else** with `ffmpeg -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le`,
  with a per-upload deadline (`--ffmpeg-timeout`) so a hostile or truncated file cannot
  hang a worker.
- **Rebuilds the form** for the engine from an allowlist of what `nemo-speech` documents
  — `model`, `language`, `response_format`, `automatic_punctuation`, `verbatim`,
  `profanity_filter`, `speech_contexts`, `prompt`, `diarization` — and **drops** the
  OpenAI/Whisper fields this endpoint has no meaning for (`temperature`,
  `no_context`, `initial_prompt`, `translate`, `word_timestamps`, …). Dropping rather
  than forwarding is what lets a Whisper-shaped client keep working.
- **Refuses by duration before decoding** — see *Long recordings*.
- **Forwards only the routes above.** `/v1/load`-style paths are 404 here; the engine is
  loopback-only regardless.
- **Stdlib only.** No pip installs, no lockfile, nothing to rebuild when the base image
  moves Python forward.

## Long recordings

Parakeet TDT is offline and full-utterance — `mode=offline head=tdt attention=full-context`
in the engine's own log — so it attends over the whole recording and encoder memory grows
**quadratically** with its length. This is the one way to kill the service, and it kills
it with `SIGKILL`, not with an error: measured against a 3 GiB limit, a 280 s clip
OOMKilled the container (exit 137, ~20 s to come back).

| recording | result | engine peak RSS |
| --- | --- | --- |
| 89.9 s | OK, 20.3 s wall | ~1 595 MiB |
| 139.4 s | OK, 39.4 s wall | ~1 784 MiB |
| 210 s | OK, 71.8 s wall | 2 209 MiB |
| **280 s** | **OOMKilled** | over the limit |
| 418 s | OOMKilled | — |

So the adapter checks duration **before decoding** — from the WAV header when it can,
from `ffprobe` otherwise — and answers `413` while the engine never sees the bytes:

```json
{"error": {"message": "recording is 280s, over the 180s this service will attempt (see README.md -> Long recordings)", "type": "invalid_request_error"}}
```

0.25 s to refuse, no memory spent, no restart. A byte cap cannot express this: 32 MiB of
Opus is roughly 110 minutes of speech.

`--max-audio-seconds 180` is a round number below the measured wall with ~25 % headroom.
Raise it only together with `mem_limit`, using the curve above; 210 s already costs 72 s
of compute, which is past most callers' patience anyway. Splitting long audio into chunks
and joining transcripts is the alternative and is deliberately not implemented — this
serves short voice notes, and there is a Whisper deployment in my lab for anything that
ever stops being one.

## Performance, on real hardware

Measured on a k3s node with an **Intel i7-8700T** (6 cores / 12 threads, 2018, AVX2 and
no AVX-512), one replica, `limits.cpu 4`, clips delivered as Ogg/Opus through the
adapter, so conversion time is included. Nine French clips (10–19 s each, Multilingual
LibriSpeech test audio, CC-BY 4.0) plus Whisper's `jfk.wav`.

| | Parakeet TDT 0.6B v3 q8_0 | whisper.cpp `base`, same clips |
| --- | --- | --- |
| word error rate, mean | **4.1 %** | 21.1 % |
| RTF, 10–19 s clips (median) | **0.149** | 0.192 |
| RTF, 139 s clip | 0.28 | 0.10 |
| peak RSS | 1 784 MiB | 511 MiB |
| CPU per 380 s of audio | 311 core-s | 212 core-s |

Read the last two rows as well as the first: **this is an accuracy win, not a throughput
win.** For a single short voice note it is also the faster one (1.7 s against 2.3 s for
11 s of audio), and it answers two notes at once where Whisper.cpp serialises on one
mutex. Published benchmarks for this model are CUDA or AVX-512 — ignore them if your CPU
is a 2018 desktop.

**Thread counts, swept rather than guessed** (same clips, HTTP workers fixed at 2,
`limits.cpu` raised to 8 so the cgroup is not the variable):

| `--asr.backend.threads` | RTF median | core-seconds | WER |
| --- | --- | --- | --- |
| 2 | 0.243 | 128 | 3.9 % |
| **4** | **0.143** | 158 | 3.9 % |
| 6 | 0.158 | 203 | 3.9 % |

Six threads is slower than four *and* costs 28 % more CPU: six physical cores, and the
encoder gains nothing from the sibling hyperthread. Accuracy never moves. Upstream's
default would be `min(8, hardware_threads)` = 8 on a 12-thread machine, which
oversubscribes any sane container quota — set this value, do not inherit it.

Cold start is ~4 s to load the weights and finish warmup natively (7 s pod-created →
Ready including a 2 s checksum of the model file), and ~10–30 s under emulation.

## Configuration

Both processes live in one image and pick their job from the command line — nothing in
the image decides which one it is.

`python3 /opt/adapter/adapter.py`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--listen` | `0.0.0.0:8080` | where the adapter publishes. The only port anyone else should reach. |
| `--upstream` | `127.0.0.1:8081` | the engine |
| `--max-upload-mb` | `32` | request body cap, refused before the body is read |
| `--max-audio-seconds` | `180` | duration cap, checked before decoding. `0` disables it. |
| `--ffmpeg-timeout` | `60` | per-upload decode deadline |
| `--upstream-timeout` | `180` | engine response deadline |
| `--workdir` | `/tmp` | scratch for the decoded WAV |

`/opt/nemo-speech/bin/nemo-speech serve` — the ones that matter:

| Flag | Shipped | Why not the default |
| --- | --- | --- |
| `--host` `--port` | `127.0.0.1` `8081` | loopback is the access control; 8081 keeps engine and adapter distinguishable in one log |
| `--asr-model` | path into the mounted volume | models are mounted, never baked; `--asr-model parakeet-tdt` would download at serve time |
| `--device` | `cpu` | the default is `auto`, which on a GPU-capable build quietly takes a device slice |
| `--asr.backend.threads` | `4` | see the sweep above. **The key is `asr.backend.threads`, not `asr.threads`** — `threads` hangs off `BackendConfig`, and the server rejects an unregistered key at startup (`unknown option: --asr.threads`), which is the one mercy of a young runtime |
| `--threads` | `2` | HTTP workers — concurrent transcriptions, a completely different knob |
| `--max-upload-mb` | `32` | matched to the adapter's, so the adapter is the one that refuses |
| `--no-ui` | on | the playground is not a thing to expose |

Runs as uid 1000, needs no writable filesystem apart from `/tmp` (so
`readOnlyRootFilesystem: true` is fine), and drops every capability. `nemo-speech
doctor` confirms what the build carries — note that even on a machine with a GPU it
lists one device:

```
NeMo-Speech.cpp 0.2.0
Features: asr diarization http integrated_vad model_pull punctuation realtime_websocket speech_translation text_normalization translation tts
Devices:
  [0] cpu
```

## The model

| | |
| --- | --- |
| file | `parakeet-tdt-0.6b-v3.q8_0.gguf` |
| URL | `huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/…` |
| size | 713 975 456 bytes (681 MiB) |
| sha256 | `e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e` |
| license | CC-BY-4.0 |
| languages | 25 European, including French and English |

The Hugging Face **revision is pinned, not `main`**, because `resolve/<branch>` changes
content silently — and a model that changes underneath a transcription comparison makes
the comparison worthless. Verify after downloading; the checksum is 1.6 s from a warm
local disk.

Two things to know before reaching for `language`:

- **This model ignores the field.** `parakeet-tdt-0.6b-v3.q8_0.gguf` carries
  `asr.rnnt.num_prompts = 0` — no language prompts to select — and in 0.2.0
  `prompt_index_for_lang()` is overridden only for RNNT models. Measured: the same
  French note with no `language`, with `fr`, with `fr-FR` and with `fr_FR` produced four
  byte-identical transcripts.
- **It follows the audio instead, which is what you wanted.** A note that changes
  language mid-recording comes back in both languages. `verbose_json` reports
  `"language": ""`, so anything that needs a language *label* — routing, logging,
  translation — needs a prompt-conditioned model such as Nemotron 3.5 ASR, not this one.

Also, no Whisper habits survive the move: `prompt` / `speech_contexts` vocabulary
boosting is accepted and ignored for TDT ("greedy CTC without an LM and Parakeet TDT
ignore it" — upstream's words), and there is no `initial_prompt` and no temperature
re-decode pass.

## Tests

```bash
python3 test_adapter.py
```

21 tests, no model, no GPU, no cluster: a stub upstream echoes what it received, and the
suite asserts the WAV arrives byte-identical, that a non-WAV upload arrives as valid
16 kHz mono PCM16, that refusals are 4xx in the right shape, and that a dead upstream is
`502` rather than an empty `200`. Two tests need ffmpeg and skip themselves without it.

**Run them inside the image too**, and do not let anyone "simplify" this away:

```bash
docker buildx build --platform=linux/amd64 -t parakeet:ci .
docker run --rm --platform linux/amd64 -v "$PWD:/t:ro" --entrypoint python3 \
    parakeet:ci -m unittest discover -s /t -p "test_*.py"
```

That command is how a real bug got caught. The release archive bundles its own
`libstdc++.so.6` and `libgcc_s.so.1`, older than Ubuntu 24.04's, and the linker prefers
`LD_LIBRARY_PATH` over system paths — **for every process in the image**. With it pointed
at `/opt/nemo-speech/lib`, ffmpeg dies at startup
(`libchromaprint.so.1: version 'GLIBCXX_3.4.32' not found`), the adapter passes the
failure through, and every Ogg note answers 400 while the engine looks perfectly healthy.
The Dockerfile therefore sets `LD_LIBRARY_PATH=` **empty**, on purpose, with a comment
asking nobody to fix it; the engine needs no help because its binaries carry
`RPATH $ORIGIN/../lib`. Green on a laptop, red in the image.

## Kubernetes

This is what the image is for, in case that is your deployment: one Pod, two containers,
the same loopback arrangement as the compose file. The parts worth copying are the
probes.

```yaml
initContainers:      # seed the model if absent, then verify it EVERY start (1.6 s)
  - name: seed-model
    # curl -z / -o .tmp + sha256sum -c + mv, so a partial download is never what loads
containers:
  - name: engine     # /opt/nemo-speech/bin/nemo-speech serve --host 127.0.0.1 …
    livenessProbe:   # exec, not httpGet: the kubelet probes the Pod IP and this
      exec:          # container binds loopback, so httpGet can never reach it
        command: ["/opt/nemo-speech/bin/nemo-speech", "health",
                  "--url", "http://127.0.0.1:8081/health", "--timeout", "10"]
    # and generous timings: two workers can be legitimately busy for a minute on a
    # long note, and a probe that gives up before that restarts a server mid-transcription
  - name: gateway    # python3 /opt/adapter/adapter.py --listen 0.0.0.0:8080
    readinessProbe:  # the engine's /ready, forwarded — no traffic before weights load
      httpGet: {path: /ready, port: 8080}
```

Plus a `ReadWriteOnce` PVC for the model (which pins the Pod to its node, so use
`strategy: Recreate`), a non-root `securityContext` (uid/gid/fsGroup 1000), and a
`NetworkPolicy` naming exactly the workloads allowed to transcribe. My deployment of it
lives in the same repo as this service's callers, at `argocd/parakeet` — this repo is the
image recipe, that one is the wiring.

## What this is not

- **Not diarised.** `diarization` is in the form allowlist because the endpoint accepts
  it, but it needs a Sortformer GGUF, which this recipe does not download.
- **Not a language identifier.** See above: `""` is what you get.
- **Not streaming.** `realtime_websocket` is compiled in, this serves files. Chunking
  long audio is an open problem here (see *Long recordings*).
- **Not authenticated.** Neither is the engine it wraps; the adapter adds no auth and
  should not be exposed beyond a trusted network. If you need a credential, the engine
  has `--api-key` — put it on the engine and have the adapter attach it, so the secret
  never reaches a client.
- **Not a benchmark harness.** The numbers above came from a script that is not in this
  repo; they are quoted with their hardware and clip set so you can discount them
  properly.

## Pinning, and upgrading

| | version | checksum / commit |
| --- | --- | --- |
| NeMo-Speech.cpp | 0.2.0 | `6a3ca369370782acee0dd155e34394ba60b920c4`, archive sha256 `f396057150f1b774935c7414fd32ebc04195a795a34120d78d8c9ecefa1b7507` |
| base image | Ubuntu 24.04 | `sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac` |
| model | parakeet-tdt-0.6b-v3 q8_0 | revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`, sha256 `e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e` |

Upgrading NeMo-Speech.cpp is three `ARG`s in the Dockerfile — version, arch, sha256 from
the release's `.sha256` asset. Expect the engine's flags to move: this is a 0.x runtime,
and `--asr.backend.threads` was already renamed once. Check `nemo-speech serve --help`
after the version bump, and re-measure the thread sweep, because "faster on average" and
"faster at four threads on six physical cores" are different claims.

Everything pinned is verified during the build (archive) or at container start (model),
so a silent upstream change fails loudly at one of those two points.

## Licence

MIT for `adapter.py`, the Dockerfile and the compose file. Nothing else here is mine to
licence: the NeMo-Speech.cpp runtime is NVIDIA's under Apache-2.0, and
`nvidia/parakeet-tdt-0.6b-v3` is NVIDIA's under CC-BY-4.0. An image built from this repo
contains both, so if you redistribute it — publicly or commercially — read those two
licences first and keep the model attribution intact.
