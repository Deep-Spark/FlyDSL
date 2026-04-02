FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ── configurable mirrors ──────────────────────────────────────────────
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG RUNNER_VERSION=2.322.0

# ── system packages ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential gcc g++ \
        git curl wget ca-certificates \
        python3 python3-pip python3-dev \
        libssl-dev zlib1g-dev libffi-dev \
        jq sudo lsb-release \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages (IR tests + build) ────────────────────────────────
RUN pip3 install --break-system-packages \
        -i ${PIP_INDEX_URL} --trusted-host ${PIP_TRUSTED_HOST} \
        pip setuptools wheel uv ninja \
        pytest 'hypothesis>=6.82.0' nanobind pybind11 numpy pybind11

# torch CPU-only (needed by tests that import flydsl.compiler)
RUN pip3 install --break-system-packages \
        torch --index-url https://download.pytorch.org/whl/cpu

# ── GitHub Actions runner ─────────────────────────────────────────────
RUN mkdir -p /actions-runner && cd /actions-runner \
    && curl --http1.1 --retry 3 -o runner.tar.gz -L \
       "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && tar xzf runner.tar.gz && rm runner.tar.gz \
    && ./bin/installdependencies.sh \
    && rm -rf /var/lib/apt/lists/*

# ── non-root runner user with passwordless sudo ───────────────────────
RUN useradd -m -d /home/runner -s /bin/bash runner \
    && echo "runner ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/runner \
    && chown -R runner:runner /actions-runner

# ── shared LLVM cache directory ───────────────────────────────────────
RUN mkdir -p /opt/flydsl/llvm-cache && chmod 1777 /opt/flydsl/llvm-cache

COPY start.sh /actions-runner/start.sh
RUN chmod +x /actions-runner/start.sh

USER runner
WORKDIR /actions-runner

ENTRYPOINT ["/actions-runner/start.sh"]
