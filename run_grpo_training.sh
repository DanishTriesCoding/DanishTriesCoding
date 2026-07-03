#!/usr/bin/env bash
# =============================================================================
# run_grpo_training.sh
#
# Full AgentEvolver GRPO self-evolution training for ARR multilingual datasets.
#
# Replicates the paper methodology:
#   1. Start env_service serving our custom environment (drop/snips/skillsbench)
#   2. Run GRPO training via veRL on that environment
#   3. The evolved model checkpoint is saved under experiments/grpo/
#   4. After training, run eval on original + multilingual variants
#
# Paper: https://arxiv.org/abs/2511.10395
# Uses verl063 conda env (veRL 0.8.0.dev, torch 2.9, vllm 0.12.0)
#
# Hardware: 2× A100 80GB (paper used 8). We use CPU offload + n=4 rollouts.
# Expected training time: ~2-4h per 20 epochs on 2 GPUs for DROP (20 tasks)
#
# Usage:
#   ENV=drop   bash scripts/run_grpo_training.sh   # train on DROP
#   ENV=snips  bash scripts/run_grpo_training.sh   # train on SNIPS
#   ENV=skillsbench bash scripts/run_grpo_training.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
AE_ROOT="$(dirname "$ROOT")/Frameworks/AgentEvolver"

# ── Config ────────────────────────────────────────────────────────────────────
ENV="${ENV:-drop}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
ENV_PORT="${ENV_PORT:-8080}"
EPOCHS="${EPOCHS:-20}"
N_GPUS="${N_GPUS:-1}"
PYTHON=/home/csalt/Desktop/.grpo_env/bin/python

# ── Prerequisites check ───────────────────────────────────────────────────────
echo "── Checking prerequisites ──────────────────────────────────────"
$PYTHON -c "import flash_attn" 2>/dev/null || {
    echo "  WARNING: flash_attn not installed in verl063."
    echo "  Training will work but be slower (standard attention)."
    echo "  Install with: pip install flash-attn==2.7.4.post1 --no-build-isolation"
    echo ""
}
$PYTHON -c "import verl; print('  verl:', verl.__version__)"
$PYTHON -c "import vllm; print('  vllm:', vllm.__version__)"

# GPU 1 is typically occupied by Gemma-3-27B (another user). Restrict to GPU 0.
# For full 2-GPU training, ensure GPU 1 is free and remove CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

LOG_DIR="$ROOT/results/grpo_logs"
EXP_DIR="$AE_ROOT/experiments/grpo_${ENV}"
mkdir -p "$LOG_DIR" "$EXP_DIR"

export PYTHONPATH="$ROOT:$AE_ROOT:${PYTHONPATH:-}"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  AgentEvolver GRPO Self-Evolution — ARR Multilingual          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo "  Environment : $ENV"
echo "  Model       : $MODEL_PATH"
echo "  GPUs        : $N_GPUS"
echo "  Epochs      : $EPOCHS"
echo "  Env service : http://localhost:$ENV_PORT"
echo ""

# ── Step 1: Start env_service ─────────────────────────────────────────────────
echo "── Step 1: Starting env_service for '$ENV' on port $ENV_PORT ─"
ENV_LOG="$LOG_DIR/env_service_${ENV}.log"

$PYTHON "$SCRIPT_DIR/launch_env_service.py" \
    --env "$ENV" \
    --portal 0.0.0.0 \
    --port "$ENV_PORT" \
    > "$ENV_LOG" 2>&1 &
ENV_PID=$!
echo "  PID: $ENV_PID  |  Log: $ENV_LOG"

trap 'echo "Stopping env_service (PID $ENV_PID)..."; kill "$ENV_PID" 2>/dev/null || true' EXIT

# Wait for env_service to start
MAX_WAIT=60
WAITED=0
echo -n "  Waiting for env_service"
until curl -sf "http://localhost:${ENV_PORT}/health" > /dev/null 2>&1 || \
      curl -sf "http://localhost:${ENV_PORT}/docs" > /dev/null 2>&1; do
    if ! kill -0 $ENV_PID 2>/dev/null; then
        echo " DIED — check $ENV_LOG"
        tail -20 "$ENV_LOG"; exit 1
    fi
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo " TIMEOUT — check $ENV_LOG"
        tail -20 "$ENV_LOG"; exit 1
    fi
    sleep 2; WAITED=$((WAITED+2)); echo -n "."
done
echo " ready in ${WAITED}s"

# ── Step 2: Prepare task files (if not already done) ──────────────────────────
echo ""
echo "── Step 2: Preparing task files ────────────────────────────────"
TASKS_DIR="$ROOT/configs/tasks"
if [[ ! -f "$TASKS_DIR/${ENV}_original_en.json" && ! -f "$TASKS_DIR/${ENV}_val_original_en.json" ]]; then
    $PYTHON "$ROOT/adapters/${ENV}_adapter.py" \
        --all-variants \
        --samples-dir "$(dirname "$ROOT")/Benchmarks/samples/${ENV}" \
        --output-dir "$TASKS_DIR"
else
    echo "  Task files already exist in $TASKS_DIR"
fi

# Determine train/val files
if [[ "$ENV" == "drop" ]]; then
    TRAIN_FILE="$TASKS_DIR/drop_train_original_en.json"
    VAL_FILE="$TASKS_DIR/drop_val_original_en.json"
else
    # SNIPS and SkillsBench have one file each
    SRC="$TASKS_DIR/${ENV}_original_en.json"
    TRAIN_FILE="$TASKS_DIR/${ENV}_train_original_en.json"
    VAL_FILE="$TASKS_DIR/${ENV}_val_original_en.json"
    if [[ ! -f "$TRAIN_FILE" ]]; then
        $PYTHON -c "
import json, random, pathlib
tasks = json.load(open('$SRC'))
random.shuffle(tasks)
n = len(tasks)
split = max(1, int(n * 0.8))
pathlib.Path('$TRAIN_FILE').write_text(json.dumps(tasks[:split], indent=2))
pathlib.Path('$VAL_FILE').write_text(json.dumps(tasks[split:], indent=2))
print(f'  Split: {split} train / {n-split} val')
"
    fi
fi
echo "  Train: $TRAIN_FILE"
echo "  Val:   $VAL_FILE"

# ── Step 3: GRPO Training ─────────────────────────────────────────────────────
echo ""
echo "── Step 3: GRPO Training (AgentEvolver) ────────────────────────"
echo "  This replicates the paper's self-evolution loop."
echo "  Qwen2.5-7B generates rollouts → env scores them → GRPO updates weights."
echo ""

TRAIN_LOG="$LOG_DIR/grpo_${ENV}_$(date +%Y%m%d_%H%M%S).log"

cd "$AE_ROOT"
$PYTHON -m agentevolver.main_ppo \
    --config-path="$ROOT/configs" \
    --config-name='grpo_2gpu' \
    env_service.env_url="http://localhost:${ENV_PORT}" \
    env_service.env_type="$ENV" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    data.train_files="$ROOT/configs/parquet/${ENV}_train_original_en.parquet" \
    data.val_files="$ROOT/configs/parquet/${ENV}_val_original_en.parquet" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.total_epochs="$EPOCHS" \
    trainer.experiment_name="${ENV}_grpo_qwen25-7b_agentevolver" \
    trainer.validation_data_dir="$EXP_DIR/validation_log" \
    trainer.rollout_data_dir="$EXP_DIR/rollout_log" \
    2>&1 | tee "$TRAIN_LOG"

echo ""
echo "── Training complete! ───────────────────────────────────────────"
echo "  Log: $TRAIN_LOG"
echo ""
echo "  Next: evaluate the evolved model on multilingual variants:"
echo "    bash scripts/run_e2e_test.sh --model <evolved_checkpoint>"
