#!/bin/bash
#SBATCH --job-name=dialogue_rating
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -A punim0478
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/gpfs/projects/punim0478/bryan/memcig-analysis
#SBATCH --output=logs/rating_%j.out
#SBATCH --error=logs/rating_%j.err
#SBATCH --mail-user=bryan.chen4@unimelb.edu.au
#SBATCH --mail-type=FAIL,BEGIN,END


set -euo pipefail

mkdir -p logs


# Activate conda env (non-interactive safe)
source /data/projects/punim0478/bryan/anaconda3/etc/profile.d/conda.sh
conda activate vllm_env

# (Optional) quick sanity checks in log
echo "=== ENV CHECK ==="
which python
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
nvidia-smi || true
echo "================="

# Ensure imports like "from prompts import ..." work
# export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Put caches somewhere with space (already set in your .bashrc, but safe)
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

# HuggingFace hub timeouts (helps on compute nodes)
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600

# Set up environment variables for vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"


# Pick a random port
PORT=$((8000 + RANDOM % 1000))
export VLLM_BASE_URL="http://localhost:$PORT/v1"
export VLLM_API_KEY="EMPTY"

echo "Starting vLLM server on port $PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 &
VLLM_PID=$!

cleanup() {
  echo "Stopping vLLM (PID $VLLM_PID)..."
  kill $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait until ready (health endpoint)
echo "Waiting for vLLM to become ready..."
for i in $(seq 1 120); do
  if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "vLLM is up."
    break
  fi
  sleep 2
done

echo "Running dialogue_rating.py..."

REPO="/data/gpfs/projects/punim0478/bryan/memcig-analysis"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

cd "$REPO/scripts"

python dialogue_rating.py \
  --model "vllm:$MODEL_PATH" \
  --backend oss \
  --input "../data/ratings/tasks_ratings_{}.json" \
  --output "../data/ratings/tasks_ratings_{}_{}.json" \
  --iterations 3 \
  --thinking-budget 1024


