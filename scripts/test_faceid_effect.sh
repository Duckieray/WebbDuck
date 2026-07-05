#!/usr/bin/env bash
# ── A/B Test: IP-Adapter FaceID Effect ──────────────────────────
# Compares outputs at multiple adapter scales against a baseline
# to verify the adapter is visibly affecting generation.
#
# Usage:
#   export REF_IMAGE="/path/to/ref/face.png"
#   export SERVER_URL="http://127.0.0.1:8010"
#   bash scripts/test_faceid_effect.sh
#
# Requires: curl, a running WebbDuck server with GPU, one ref image.

set -euo pipefail

SERVER="${SERVER_URL:-http://127.0.0.1:8010}"
REF="${REF_IMAGE:-}"
OUTDIR="outputs/faceid_ab_test_$(date +%s)"
PROMPT="portrait photo of a woman, neutral background, looking at camera"
SEED="${SEED:-42}"
MODEL="${MODEL:-}"
CHKP="${CHECKPOINT:-}"

mkdir -p "$OUTDIR"

if [ -z "$REF" ]; then
    echo "ERROR: REF_IMAGE not set. Point it to a face reference image."
    echo "  export REF_IMAGE=/path/to/ref.png"
    exit 1
fi
if [ ! -f "$REF" ]; then
    echo "ERROR: REF_IMAGE '$REF' not found."
    exit 1
fi

echo "Server : $SERVER"
echo "Ref    : $REF"
echo "Prompt : $PROMPT"
echo "Seed   : $SEED"
echo "Outdir : $OUTDIR"
echo ""

# Detect the first available model from the server
_model_json=$(curl -s "$SERVER/models")
MODEL=$(echo "$_model_json" | python3 -c "
import sys,json
data=json.load(sys.stdin)
if isinstance(data,list):
    for m in data:
        name=m.get('name','') or m.get('id','')
        if name: print(name); break
elif isinstance(data,dict):
    for k in ('models', 'checkpoints'):
        val=data.get(k,[])
        if val: print(val[0].get('name','') or val[0].get('id','') or list(val.keys())[0]); break
")
if [ -z "$MODEL" ]; then
    echo "ERROR: could not detect a model from $SERVER/models"
    echo "Set CHECKPOINT env var manually."
    exit 1
fi
echo "Model  : $MODEL"
echo ""

do_gen() {
    local label="$1"
    local scale="$2"
    local outfile="$OUTDIR/${label}.png"

    # Build identity_adapter JSON
    local id_json='{"enabled":false}'
    if [ "$scale" != "none" ]; then
        id_json=$(cat <<ENDJSON
{
    "enabled": true,
    "type": "faceid_sdxl",
    "repo": "h94/IP-Adapter-FaceID",
    "adapter_weight": "ip-adapter-faceid_sdxl.bin",
    "embedder": "buffalo_l",
    "adapter_scale": $scale,
    "reference_images": ["$REF"],
    "reference_mode": "primary_only"
}
ENDJSON
)
    fi

    echo "  [$label] scale=$scale …"
    resp=$(curl -s -X POST "$SERVER/test" \
        -F "prompt=$PROMPT" \
        -F "base_model=$MODEL" \
        -F "seed=$SEED" \
        -F "steps=28" \
        -F "cfg=7.0" \
        -F "width=1024" \
        -F "height=1024" \
        -F 'scheduler=UniPC' \
        -F "identity_adapter=$id_json" \
        -F 'wait_for_result=true')

    # Extract image URL from response
    img_url=$(echo "$resp" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    imgs=d.get('images',[])
    if imgs: print(imgs[0])
except: pass
")
    if [ -z "$img_url" ]; then
        echo "  ERROR: no image in response. Response:"
        echo "$resp" | python3 -m json.tool 2>/dev/null || echo "$resp"
        return 1
    fi

    # Download image
    curl -s -o "$outfile" "${SERVER}/${img_url}"
    echo "  -> saved $outfile"
    echo ""
}

echo "=== Baseline (no adapter) ==="
do_gen "00_baseline" "none"

echo "=== FaceID scale=0.5 ==="
do_gen "01_scale_05" "0.5"

echo "=== FaceID scale=1.0 ==="
do_gen "02_scale_10" "1.0"

echo "=== FaceID scale=1.5 ==="
do_gen "03_scale_15" "1.5"

echo "=== Done ==="
echo "Outputs in: $OUTDIR"
echo ""
echo "To verify visible effect, compare:"
echo "  $OUTDIR/00_baseline.png  (no identity conditioning)"
echo "  $OUTDIR/03_scale_15.png  (strongest identity conditioning)"
echo ""
echo "At scale >=1.0, the subject should visibly resemble the reference face."
echo "If 00_baseline and 03_scale_15 look identical, the adapter is NOT working."
