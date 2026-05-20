#!/bin/bash
set -euo pipefail

source $HOME/miniforge3/etc/profile.d/conda.sh

# ===== 配置 =====
CKPT="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA/results/Checkpoints/qwengr00t_qwen3vl_4b_calvin_abcd_max/checkpoints/steps_15000_pytorch_model.pt"
CALVIN_DATA="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/calvin/dataset/task_D_D"
CALVIN_CONF="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/calvin/calvin_models/conf"
STARVLA_HOME="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA"
RESULTS_BASE="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/eval_results"
BASE_PORT=5794
NUM_GPU=1
# ================

cd "$STARVLA_HOME"
export PYTHONPATH="${STARVLA_HOME}:${PYTHONPATH:-}"

DATE_SUFFIX=$(date +%m%d_%H%M)
RUN_ID="eval_qwengr00t_qwen3vl_${DATE_SUFFIX}"
RUN_DIR="${RESULTS_BASE}/${RUN_ID}"
SHARD_DIR="${RUN_DIR}/shards"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"

mkdir -p "$SHARD_DIR" "$LOG_DIR" "$RESULT_DIR"

echo "============================================"
echo "  Run ID:    ${RUN_ID}"
echo "  Run Dir:   ${RUN_DIR}"
echo "  CKPT:      ${CKPT}"
echo "  Num GPU:   ${NUM_GPU}"
echo "============================================"

# ---- 切分 eval_sequences.json ----
conda activate starVLA
python3 -c "
import json
with open('examples/calvin/eval_files/eval_sequences.json') as f:
    seqs = json.load(f)
shard_size = (len(seqs) + ${NUM_GPU} - 1) // ${NUM_GPU}
for i in range(${NUM_GPU}):
    shard = seqs[i*shard_size:(i+1)*shard_size]
    with open(f'${SHARD_DIR}/shard_{i}.json', 'w') as f:
        json.dump(shard, f)
    print(f'Shard {i}: {len(shard)} sequences')
"

SERVER_PIDS=()
EVAL_PIDS=()

cleanup() {
    echo "Cleaning up..."
    for pid in "${EVAL_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${SERVER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- 启动 Policy Server ----
echo ""
echo "=== Starting Policy Servers ==="
for i in $(seq 0 $((NUM_GPU - 1))); do
    port=$((BASE_PORT + i))
    conda activate starVLA
    CUDA_VISIBLE_DEVICES=$i python deployment/model_server/server_policy.py \
        --ckpt_path "$CKPT" --port "$port" --use_bf16 \
        > "${LOG_DIR}/server_gpu${i}.log" 2>&1 &
    SERVER_PIDS+=($!)
    echo "  GPU $i, port $port, PID ${SERVER_PIDS[$i]}"
done

# ---- 等待所有 Server 就绪 ----
echo ""
echo "=== Waiting for servers ==="
for i in $(seq 0 $((NUM_GPU - 1))); do
    port=$((BASE_PORT + i))
    while ! grep -q "listening" "${LOG_DIR}/server_gpu${i}.log" 2>/dev/null; do
        if ! kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
            echo "ERROR: Server $i died. Check ${LOG_DIR}/server_gpu${i}.log"
            exit 1
        fi
        sleep 2
    done
    echo "  Server $i ready on port $port"
done

# ---- 启动 Calvin 评测 ----
echo ""
echo "=== Starting Calvin Eval ==="
for i in $(seq 0 $((NUM_GPU - 1))); do
    port=$((BASE_PORT + i))
    conda activate calvin_venv
    python ./examples/calvin/eval_files/eval_calvin.py \
        --args.pretrained-path "$CKPT" \
        --args.unnorm-key franka \
        --args.host 127.0.0.1 \
        --args.port "$port" \
        --args.dataset_path "$CALVIN_DATA" \
        --args.calvin_config_path "$CALVIN_CONF" \
        --args.eval_sequences_path "${SHARD_DIR}/shard_${i}.json" \
        --args.eval_log_dir "${RESULT_DIR}/shard_${i}" \
        > "${LOG_DIR}/eval_shard${i}.log" 2>&1 &
    EVAL_PIDS+=($!)
    echo "  Shard $i, port $port, PID ${EVAL_PIDS[$i]}"
done

# ---- 等待所有评测完成 ----
echo ""
echo "=== Waiting for evaluation ==="
FAILED_SHARDS=0
for i in $(seq 0 $((NUM_GPU - 1))); do
    wait "${EVAL_PIDS[$i]}" && echo "  Shard $i DONE" || { echo "  Shard $i FAILED"; FAILED_SHARDS=$((FAILED_SHARDS + 1)); }
done

trap - EXIT INT TERM
cleanup

# ---- 汇总结果 ----
echo ""
echo "============================================"
echo "       FINAL RESULTS"
echo "============================================"
echo ""

conda activate starVLA
python3 "${STARVLA_HOME}/examples/calvin/eval_files/aggregate_live_metrics.py" \
    --run-dir "${RUN_DIR}" --num-shards "${NUM_GPU}"

echo ""
echo "Run directory: ${RUN_DIR}"
echo "  logs/       - server + eval logs"
echo "  shards/     - input sequence splits"
echo "  results/    - per shard eval output"
echo "  aggregate.json - final aggregated results"
