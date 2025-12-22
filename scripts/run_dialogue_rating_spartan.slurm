#!/bin/bash
#SBATCH --job-name=dialogue_rating
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/mingbin/punim0478/memcig-analysis
#SBATCH --output=logs/rating_%j.out
#SBATCH --error=logs/rating_%j.err


set -euo pipefail

mkdir -p logs

# Load necessary modules
module purge
module load Python/3.11.3
module load CUDA/12.2.0

# Activate virtual environment
# Adjust the path to your actual venv
source ~/venvs/myenv/bin/activate

# Put caches somewhere with space (adjust to your project filesystem)
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

# Set up environment variables for vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"


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

# Run the rating script
# Note: We pass "vllm:$MODEL_PATH" to tell the script to use the vLLM backend logic
# but since we set VLLM_BASE_URL, the oss_utility should pick it up.
# Actually, looking at oss_utility.py:
# if model.startswith("vllm:"): backend="vllm", raw_model=model[5:]
# So we should pass --model "vllm:$MODEL_PATH" --backend oss

echo "Running dialogue_rating.py..."
python scripts/dialogue_rating.py \
    --model "vllm:$MODEL_PATH" \
    --backend oss \
    --input "data/ratings/tasks_ratings_{}.json" \
    --output "data/ratings/tasks_ratings_{}_{}.json" \
    --iterations 1 \
    --thinking-budget 1024

