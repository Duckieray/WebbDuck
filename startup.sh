#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${WEBBDUCK_CONDA_ENV:-webbduck}"
OUTPUT_DIR="${WEBBDUCK_OUTPUT_DIR:-$SCRIPT_DIR/outputs}"
PORT="${WEBBDUCK_PORT:-8010}"
USE_CONDA=1
EXTRA_ARGS=()

usage() {
  cat <<USAGE
Usage: ./startup.sh [options] [-- extra run.py args]

Starts WebbDuck from this repository directory.

Options:
  -e, --env NAME        Conda environment name (default: $CONDA_ENV)
  -o, --output DIR      Output directory (default: $OUTPUT_DIR)
  -p, --port PORT       Server port (default: $PORT)
      --dtype DTYPE     Set WEBBDUCK_DTYPE (float16|bfloat16|float32)
      --device DEVICE   Set WEBBDUCK_DEVICE (cuda|cpu)
      --no-conda        Use current shell Python without conda activation
  -h, --help            Show this help and exit

Examples:
  ./startup.sh
  ./startup.sh --env webbduck --output ./outputs --port 8020
  ./startup.sh --dtype float16 --port 8020
USAGE
}

fail() {
  echo "[startup] ERROR: $*" >&2
  exit 1
}

info() {
  echo "[startup] $*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--env)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      CONDA_ENV="$2"
      shift 2
      ;;
    -o|--output)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -p|--port)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      PORT="$2"
      shift 2
      ;;
    --dtype)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      export WEBBDUCK_DTYPE="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      export WEBBDUCK_DEVICE="$2"
      shift 2
      ;;
    --no-conda)
      USE_CONDA=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      fail "Unknown argument: $1 (use --help)"
      ;;
  esac
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  fail "Invalid port: $PORT"
fi

mkdir -p "$OUTPUT_DIR"

: "${HF_HUB_DISABLE_PROGRESS_BARS:=1}"
: "${TQDM_DISABLE:=1}"
export HF_HUB_DISABLE_PROGRESS_BARS
export TQDM_DISABLE

if (( USE_CONDA )); then
  command -v conda >/dev/null 2>&1 || fail "Conda is not available in PATH"

  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  [[ -n "$CONDA_BASE" ]] || fail "Unable to determine conda base path"

  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || fail "Failed to activate conda env: $CONDA_ENV"
fi

command -v python >/dev/null 2>&1 || fail "python not found after environment setup"

info "Repository: $SCRIPT_DIR"
info "Conda env: ${CONDA_DEFAULT_ENV:-<current shell>}"
info "Output: $OUTPUT_DIR"
info "Port: $PORT"
if [[ -n "${WEBBDUCK_DTYPE:-}" ]]; then
  info "WEBBDUCK_DTYPE=$WEBBDUCK_DTYPE"
fi
if [[ -n "${WEBBDUCK_DEVICE:-}" ]]; then
  info "WEBBDUCK_DEVICE=$WEBBDUCK_DEVICE"
fi

exec python run.py --output "$OUTPUT_DIR" --port "$PORT" "${EXTRA_ARGS[@]}"
