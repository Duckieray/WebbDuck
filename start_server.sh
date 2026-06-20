#!/bin/bash
# Start WebbDuck server persistently with conda environment
# Usage: ./start_server.sh

cd /var/home/duckie/git/WebbDuck

setsid bash -c '
source /home/duckie/miniconda3/etc/profile.d/conda.sh
conda activate webbduck
export WEBBDUCK_MODELS_DIR="/run/media/duckie/Leia/models"
export WEBBDUCK_HF_CACHE_DIR="/run/media/duckie/Leia/huggingface"
export WEBBDUCK_OUTPUT_DIR="/home/duckie/git/WebbDuck/outputs"
nohup python run.py --output /home/duckie/git/WebbDuck/outputs --port 8010 --models /run/media/duckie/Leia/models --hf-cache /run/media/duckie/Leia/huggingface > /tmp/webbduck_server.log 2>&1 &
' >/dev/null 2>&1 &

echo "WebbDuck server starting on http://localhost:8010"
echo "Check status: curl http://localhost:8010/health"
echo "View logs: tail -f /tmp/webbduck_server.log"
