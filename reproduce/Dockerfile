# Hermetic environment for independently re-grading the published MLE-bench
# Lite-22 submissions with OpenAI's unmodified mlebench grader.
#
# This pins the interpreter + numerical stack to the versions our medals were
# graded under (see reproduce/requirements.lock — pandas 3.0.3 is load-bearing).
# The image does NOT bundle competition data or Kaggle credentials: you supply
# those at run time (rule-acceptance is a manual, per-account Kaggle step — see
# reproduce/VERIFY.md). The engine that produced the submissions is proprietary
# and is not part of this repository; re-grading needs only OpenAI's grader.
#
# Build (from repo root):
#   docker build -f reproduce/Dockerfile -t mlebench-verify .
#
# Run (mount your Kaggle creds + a data cache):
#   docker run --rm -it \
#     -v "$HOME/.kaggle:/root/.kaggle:ro" \
#     -v mlebench-data:/root/.cache/mle-bench/data \
#     mlebench-verify
#   # then, inside the container:
#   mlebench prepare -c nomad2018-predict-transparent-conductors
#   mlebench grade-sample \
#     solutions/nomad2018-predict-transparent-conductors/submission.csv \
#     nomad2018-predict-transparent-conductors
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
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r reproduce/requirements.lock

# Copy the repo (ledger, solutions, verification tooling).
COPY . .

ENTRYPOINT ["bash"]
