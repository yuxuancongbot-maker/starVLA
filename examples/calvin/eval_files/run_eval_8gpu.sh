#!/bin/bash
set -euo pipefail

source $HOME/miniforge3/etc/profile.d/conda.sh

# ===== 配置 =====
# [必须改] 改成你自己的 checkpoint，避免评测到别人的模型
CKPT="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA/results/Checkpoints/QwenPI_v3_qwen35_2b_calvin_abcd/final_model/pytorch_model.pt"
CALVIN_DATA="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/calvin/dataset/task_D_D"
CALVIN_CONF="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/calvin/calvin_models/conf"
# [必须改] 端口不能和别人重复, 如果别人用 5694-5701，你可以用 5794-5801 或 5894-5901
BASE_PORT=5794
NUM_GPU=8
# [必须改]  starVLA 路径
STARVLA_HOME="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/starVLA"
# ================

cd "$STARVLA_HOME"
export PYTHONPATH="${STARVLA_HOME}:${PYTHONPATH:-}"

# ---- 切分 eval_sequences.json ----
conda activate starVLA
SHARD_DIR="/tmp/calvin_eval_shards"
mkdir -p "$SHARD_DIR"
python3 -c "
import json
with open('examples/calvin/eval_files/eval_sequences.json') as f:
    seqs = json.load(f)
shard_size = (len(seqs) + $NUM_GPU - 1) // $NUM_GPU
for i in range($NUM_GPU):
    shard = seqs[i*shard_size:(i+1)*shard_size]
    with open(f'${SHARD_DIR}/shard_{i}.json', 'w') as f:
        json.dump(shard, f)
    print(f'Shard {i}: {len(shard)} sequences')
"

SERVER_PIDS=()
EVAL_PIDS=()
#得改！！！！！！！！！！！！！
LOG_DIR="/inspire/qb-ilm2/project/26summer-camp-10/26220218/congyuxuan/logs"
mkdir -p "$LOG_DIR"

cleanup() {
    echo "Cleaning up..."
    for pid in "${EVAL_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${SERVER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- 启动 8 个 Policy Server ----
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

# ---- 启动 8 个 Calvin 评测 ----
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
        --args.eval_log_dir "tmp/calvin/shard_${i}" \
        > "${LOG_DIR}/eval_shard${i}.log" 2>&1 &
    EVAL_PIDS+=($!)
    echo "  Shard $i, port $port, PID ${EVAL_PIDS[$i]}"
done

# ---- 等待所有评测完成 ----
echo "=== Waiting for evaluation ==="
for i in $(seq 0 $((NUM_GPU - 1))); do
    wait "${EVAL_PIDS[$i]}" && echo "  Shard $i DONE" || echo "  Shard $i FAILED"
done

# ---- 汇总结果 ----
echo ""
echo "=== Results ==="
for i in $(seq 0 $((NUM_GPU - 1))); do
    echo "Shard $i:"
    grep -E "success|Avg" "${LOG_DIR}/eval_shard${i}.log" | tail -5 || echo "  (no summary found)"
done
