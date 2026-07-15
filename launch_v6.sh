#!/bin/bash
# Launch train_qlora_v6.py with DDP on 4× RX 6600 XT
# Usage: bash launch_v6.sh [OPTIONS]
#
# Options:
#   --resume      Resume from latest checkpoint (auto-detected)
#   --4bit/--qlora  Use 4-bit NF4 quantization (saves ~4GB VRAM per GPU)
#   --test        Quick 3-step test run (no actual training)
#
# Examples:
#   bash launch_v6.sh --test              # Quick test (3 steps)
#   bash launch_v6.sh                     # Full training in fp16
#   bash launch_v6.sh --4bit              # Full training in 4-bit (if fp16 OOMs)
#   bash launch_v6.sh --4bit --test       # Quick test in 4-bit mode

set -euo pipefail

cd /home/nodeadmin/trading-llm

# ROCm environment
export HSA_OVERRIDE_GFX_VERSION="10.3.0"
export HSA_ENABLE_SDMA="0"
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
export NCCL_P2P_LEVEL="SYS"
export NCCL_DEBUG="WARN"
export NCCL_IB_DISABLE="1"
export NCCL_NET="Socket"
export HIP_VISIBLE_DEVICES="0,1,2,3"

# Parse arguments
EXTRA_ARGS=""
MODE="fp16"
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            echo "[TEST MODE] Running 3 steps only..."
            EXTRA_ARGS="--max_steps 3"
            shift
            ;;
        --4bit|--qlora)
            MODE="4bit"
            shift
            ;;
        --resume)
            echo "[RESUME] Will auto-detect latest checkpoint..."
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [[ "$MODE" == "4bit" ]]; then
    export USE_4BIT="1"
    echo "[4-BIT MODE] Using NF4 quantization (QLoRA) — ~2GB per GPU instead of ~6GB"
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOGFILE="training_run_v6_${MODE}_${TIMESTAMP}.log"

echo "============================================"
echo "Launching DDP training on 4 GPUs (${MODE} mode)"
echo "Log: ${LOGFILE}"
echo "============================================"

# Method 1: torchrun (recommended for ROCm)
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_id=1 \
    --rdzv_backend=cfile \
    --rdzv_endpoint=/tmp/torchrun_v6_${TIMESTAMP} \
    train_qlora_v6.py \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOGFILE}"

echo ""
echo "Training complete. Log saved to: ${LOGFILE}"

# Method 2: accelerate (alternative, uncomment if torchrun has issues)
# accelerate launch \
#     --config_file accelerate_ddp_config.yaml \
#     train_qlora_v6.py \
#     ${EXTRA_ARGS} \
#     2>&1 | tee "${LOGFILE}"