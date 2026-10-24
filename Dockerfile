# syntax=docker/dockerfile:1
# Speech-to-text image: NVIDIA's NeMo-Speech.cpp CPU runtime (Parakeet TDT 0.6B v3)
# plus the two things it needs around it — ffmpeg, because the engine only decodes
# WAV and everything real arrives as Opus in Ogg, and python3, because adapter.py is
# what runs ffmpeg.
#
# Why a release archive and not docker/Dockerfile from upstream: that Dockerfile's
# builder stage is `nvcr.io/nvidia/cuda:<ver>-devel-ubuntu24.04` (~8 GB) whatever it
# builds, because one Dockerfile serves every backend. A CPU-only box does not need
# an 8 GB toolchain to produce the same `nemo-speech-<ver>-linux-x86_64-cpu` artifact
# NVIDIA publishes, signs and tests. The archive *is* the official CPU build; it is
# pinned by sha256 below, so this file cannot silently pick up a different binary.
#
# Build (amd64 only — the runtime ships linux-x86_64. On Apple Silicon the platform
# flag is what makes the failure happen here instead of at container start):
#
#   docker buildx build --platform=linux/amd64 \
#       -t ghcr.io/vincentkelleher/nemo-speech-parakeet:0.2.0 --load .
#
# Upstream: https://github.com/NVIDIA/NeMo-Speech.cpp  (v0.2.0 = 6a3ca369)
# Model:    https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3  (mounted at run time,
#           never baked into an image — see README.md -> The model)

FROM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac

ARG NEMO_SPEECH_VERSION=0.2.0
ARG NEMO_SPEECH_ARCH=linux-x86_64-cpu
# sha256 of the release archive, from the matching .sha256 asset next to it on the
# GitHub release (also reproduced in SHA256SUMS). A build that drifts fails here, at
# the step that can say why, rather than at model load three layers later.
ARG NEMO_SPEECH_SHA256=f396057150f1b774935c7414fd32ebc04195a795a34120d78d8c9ecefa1b7507
ARG NEMO_SPEECH_URL=https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        python3 \
    && rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    archive="/tmp/nemo-speech.tar.gz"; \
    curl -fsSL --retry 5 --retry-delay 5 --output "$archive" \
        "${NEMO_SPEECH_URL}/v${NEMO_SPEECH_VERSION}/nemo-speech-${NEMO_SPEECH_VERSION}-${NEMO_SPEECH_ARCH}.tar.gz"; \
    echo "${NEMO_SPEECH_SHA256}  $archive" | sha256sum -c -; \
    mkdir -p /opt/nemo-speech; \
    tar -xzf "$archive" -C /opt/nemo-speech --strip-components=1; \
    rm -f "$archive"; \
    # Headers, CMake packages and licenses are SDK material; the runtime needs bin/
    # and lib/ and nothing else. Roughly halves the image.
    rm -rf /opt/nemo-speech/include /opt/nemo-speech/lib/cmake; \
    /opt/nemo-speech/bin/nemo-speech --version

COPY adapter.py /opt/adapter/adapter.py

# Set to empty, deliberately, and do not "fix" it by pointing it at /opt/nemo-speech/lib.
# The release archive bundles its own libstdc++.so.6 and libgcc_s.so.1 (older than
# Ubuntu 24.04's, so the runtime runs on older distros), and the dynamic linker takes
# LD_LIBRARY_PATH before the system paths — for *every* process in the image. With it
# set, ffmpeg dies on startup:
#   libchromaprint.so.1: version `GLIBCXX_3.4.32' not found
# which is test_adapter.py failing inside this image and, in production, every voice
# note answering 400. The engine needs no help: its binaries carry RPATH
# $ORIGIN/../lib and resolve their own copies either way (verified with ldd). If a
# future build ever needs the variable, scope it to the engine — a wrapper around
# /opt/nemo-speech/bin/nemo-speech — never to the image.
ENV LD_LIBRARY_PATH= \
    # readOnlyRootFilesystem is on wherever this runs, so nothing may want a writable
    # home. /tmp is the scratch both processes get.
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1

# uid 1000 already exists in the base image as `ubuntu`; run as it, and give any
# mounted model directory the same ownership.
USER 1000:1000
WORKDIR /tmp

EXPOSE 8080

# One image, two jobs — the recognizer and the adapter are separate processes (see
# README.md -> Running it), and neither needs the other's command line:
#   engine  -> /opt/nemo-speech/bin/nemo-speech serve ...   (loopback only)
#   gateway -> python3 /opt/adapter/adapter.py ...          (the published port)
CMD ["/opt/nemo-speech/bin/nemo-speech", "--help"]
