# DFL GPU runtime — Python 3.8 + CUDA 11.0 + cuDNN 8
#
# Python 3.7 EOL June 2023; deadsnakes dropped it from focal.
# Python 3.8 is Ubuntu 20.04's native Python and has official TF 2.4.0 wheels.
# TF 2.4.0 requires CUDA 11.0 and cuDNN 8.0.
# Host driver must be >= 450.80 (CUDA 11.0 minimum).
# Pass GPU via: --gpus all  or compose deploy.resources block.
#
# Container layout:
#   /app/          DFL source code (main.py, facelib/, core/, models/, etc.)
#   /workspace/    host-mounted persistent data — never baked into image
#
# All pipeline data (data_src, data_dst, model, merged) lives on the host.

FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

# Suppress interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ────────────────────────────────────────────────────────────
# python3.8 is Ubuntu 20.04's default — no PPA needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Make python/pip resolve to 3.8
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3       1

# ── DFL Python dependencies ────────────────────────────────────────────────────
# Copied first so this layer is cached unless the requirements file changes.
COPY requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── DFL source code ────────────────────────────────────────────────────────────
COPY . /app/

# ── Workspace mount point ──────────────────────────────────────────────────────
# Actual data lives on the host; this is an empty target baked into the image.
RUN mkdir -p /workspace

WORKDIR /app

# Strip Windows CRLF so the shebang line works on Linux
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && \
    chmod +x /app/docker-entrypoint.sh

ENV DFL_WORKSPACE=/workspace

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["bash"]
