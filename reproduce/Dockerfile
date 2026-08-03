# Hermetic environment for reproducing MLE-bench Lite-22 Class-A medals.
#
# This pins the interpreter + numerical stack to the versions our medals were
# graded under (see reproduce/requirements.lock — pandas 3.0.3 is load-bearing).
# The image does NOT bundle competition data or Kaggle credentials: you supply
# those at run time (rule-acceptance is a manual, per-account Kaggle step — see
# reproduce/VERIFY.md).
#
# Build (from repo root):
#   docker build -f reproduce/Dockerfile -t mlebench-verify .
#
# Run (mount your Kaggle creds + a data cache; GEMINI/web keys optional):
#   docker run --rm -it \
#     -v "$HOME/.kaggle:/root/.kaggle:ro" \
#     -v "$PWD/bench/runs:/app/bench/runs" \
#     -v mlebench-data:/root/.cache/mle-bench/data \
#     mlebench-verify reproduce/reproduce.sh --tasks "nomad2018-predict-transparent-conductors"
#
# CPU-only. GPU (Class-B) medals are captured separately in reproduce/gpu/.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLEBENCH_DATA_DIR=/root/.cache/mle-bench/data

# Build tools + git (mlebench installs from git; some wheels need a compiler).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the pinned stack first (better layer caching). CPU torch wheels come
# from the PyTorch CPU index.
COPY reproduce/requirements.lock reproduce/requirements.lock
COPY bench/requirements.txt bench/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r reproduce/requirements.lock \
 && pip install -r bench/requirements.txt

# Copy the repo (the agent + harness + grader wrapper).
COPY . .

ENTRYPOINT ["bash"]
