{ pkgs ? import <nixpkgs> { } }:
let
  src = ./.;

  launcher = pkgs.writeShellApplication {
    name = "webbduck";
    runtimeInputs = with pkgs; [
      bash
      coreutils
      findutils
      gnugrep
      gawk
      python310
    ];
    text = ''
      set -euo pipefail

      SRC="${src}"
      CACHE_BASE="${XDG_CACHE_HOME:-$HOME/.cache}/webbduck"
      VENV_DIR="$CACHE_BASE/venv"
      REQ_FILE="${WEBBDUCK_REQUIREMENTS_FILE:-$SRC/requirements.txt}"
      REQ_HASH_FILE="$CACHE_BASE/requirements.sha256"
      OUTPUT_DIR="${WEBBDUCK_OUTPUT_DIR:-$PWD/outputs}"
      PORT="${WEBBDUCK_PORT:-8010}"

      mkdir -p "$CACHE_BASE"

      if [ ! -d "$VENV_DIR" ]; then
        echo "[webbduck:nix] Creating virtualenv at $VENV_DIR"
        python -m venv "$VENV_DIR"
      fi

      # shellcheck disable=SC1091
      . "$VENV_DIR/bin/activate"

      if [ ! -f "$REQ_FILE" ]; then
        echo "[webbduck:nix] ERROR: requirements file not found: $REQ_FILE" >&2
        exit 1
      fi

      REQ_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
      PREV_HASH=""
      if [ -f "$REQ_HASH_FILE" ]; then
        PREV_HASH="$(cat "$REQ_HASH_FILE" || true)"
      fi

      if [ "$REQ_HASH" != "$PREV_HASH" ]; then
        echo "[webbduck:nix] Installing/updating Python deps from $REQ_FILE"
        export PIP_DISABLE_PIP_VERSION_CHECK=1
        python -m pip install --upgrade pip setuptools wheel
        python -m pip install -r "$REQ_FILE"
        echo "$REQ_HASH" > "$REQ_HASH_FILE"
      fi

      mkdir -p "$OUTPUT_DIR"
      exec python "$SRC/run.py" --output "$OUTPUT_DIR" --port "$PORT" "$@"
    '';
  };
in
pkgs.stdenvNoCC.mkDerivation {
  pname = "webbduck-launcher";
  version = "0.1.0";
  dontUnpack = true;
  installPhase = ''
    mkdir -p "$out/bin"
    ln -s "${launcher}/bin/webbduck" "$out/bin/webbduck"
  '';
  meta = with pkgs.lib; {
    description = "WebbDuck launcher for nix run";
    license = licenses.asl20;
    platforms = platforms.linux;
    mainProgram = "webbduck";
  };
}

