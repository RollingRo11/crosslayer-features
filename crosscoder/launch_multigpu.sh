#!/bin/bash

# Multi-GPU Crosscoder Training Launcher
# This script provides convenient presets for common training configurations

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo -e "${BLUE}Multi-GPU Crosscoder Training Launcher${NC}"
    echo ""
    echo "Usage: $0 [PRESET] [OPTIONS]"
    echo ""
    echo "PRESETS:"
    echo "  gpt2-4gpu       GPT-2 training on 4 GPUs"
    echo "  gpt2-8gpu       GPT-2 training on 8 GPUs"
    echo "  gemma-4gpu      Gemma-2-2B training on 4 GPUs"
    echo "  gemma-8gpu      Gemma-2-2B training on 8 GPUs (recommended)"
    echo "  gemma-8gpu-large  Gemma-2-2B with large ae_dim (131k) on 8 GPUs with CPU offload"
    echo "  custom          Custom configuration (requires manual OPTIONS)"
    echo ""
    echo "OPTIONS (override preset defaults):"
    echo "  --num_gpus N            Number of GPUs to use"
    echo "  --ae_dim N              Autoencoder dimension"
    echo "  --batch_size N          Batch size per GPU"
    echo "  --grad_accum N          Gradient accumulation steps"
    echo "  --lr FLOAT              Learning rate"
    echo "  --steps N               Training steps"
    echo "  --cpu_offload           Enable FSDP CPU offloading"
    echo "  --no_amp                Disable automatic mixed precision"
    echo ""
    echo "Examples:"
    echo "  $0 gemma-8gpu"
    echo "  $0 gemma-4gpu --batch_size 512"
    echo "  $0 custom --num_gpus 2 --ae_dim 32768 --batch_size 1024"
}

# Parse preset
PRESET=${1:-""}
shift || true

# Default values
NUM_GPUS=8
MODEL="gemma2-2b"
AE_DIM=32768
BATCH_SIZE=256
GRAD_ACCUM=4
LR=5e-5
STEPS=50000
WARMUP_STEPS=5000
CPU_OFFLOAD=""
AMP_FLAG=""
EXTRA_ARGS=""

# Apply preset
case "$PRESET" in
    gpt2-4gpu)
        NUM_GPUS=4
        MODEL="gpt2"
        AE_DIM=32768
        BATCH_SIZE=512
        GRAD_ACCUM=2
        STEPS=30000
        WARMUP_STEPS=3000
        echo -e "${GREEN}Using preset: GPT-2 on 4 GPUs${NC}"
        ;;
    gpt2-8gpu)
        NUM_GPUS=8
        MODEL="gpt2"
        AE_DIM=32768
        BATCH_SIZE=512
        GRAD_ACCUM=2
        STEPS=30000
        WARMUP_STEPS=3000
        echo -e "${GREEN}Using preset: GPT-2 on 8 GPUs${NC}"
        ;;
    gemma-4gpu)
        NUM_GPUS=4
        MODEL="gemma2-2b"
        AE_DIM=32768
        BATCH_SIZE=256
        GRAD_ACCUM=8
        STEPS=50000
        WARMUP_STEPS=5000
        echo -e "${GREEN}Using preset: Gemma-2-2B on 4 GPUs${NC}"
        ;;
    gemma-8gpu)
        NUM_GPUS=8
        MODEL="gemma2-2b"
        AE_DIM=32768
        BATCH_SIZE=256
        GRAD_ACCUM=4
        STEPS=50000
        WARMUP_STEPS=5000
        echo -e "${GREEN}Using preset: Gemma-2-2B on 8 GPUs${NC}"
        ;;
    gemma-8gpu-large)
        NUM_GPUS=8
        MODEL="gemma2-2b"
        AE_DIM=131072
        BATCH_SIZE=128
        GRAD_ACCUM=8
        STEPS=100000
        WARMUP_STEPS=5000
        CPU_OFFLOAD="--fsdp_cpu_offload"
        echo -e "${GREEN}Using preset: Gemma-2-2B Large (131k dim) on 8 GPUs with CPU offload${NC}"
        ;;
    custom)
        echo -e "${YELLOW}Using custom configuration - please specify all required options${NC}"
        ;;
    "")
        echo -e "${RED}Error: No preset specified${NC}"
        print_usage
        exit 1
        ;;
    *)
        echo -e "${RED}Error: Unknown preset '$PRESET'${NC}"
        print_usage
        exit 1
        ;;
esac

# Parse additional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --ae_dim)
            AE_DIM="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --cpu_offload)
            CPU_OFFLOAD="--fsdp_cpu_offload"
            shift
            ;;
        --no_amp)
            AMP_FLAG="--no_amp"
            shift
            ;;
        *)
            # Pass through any other arguments
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Calculate effective batch size
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

# Print configuration
echo ""
echo -e "${BLUE}Training Configuration:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Model:                    ${YELLOW}$MODEL${NC}"
echo -e "AE Dimension:             ${YELLOW}$AE_DIM${NC}"
echo -e "Number of GPUs:           ${YELLOW}$NUM_GPUS${NC}"
echo -e "Batch size per GPU:       ${YELLOW}$BATCH_SIZE${NC}"
echo -e "Gradient accumulation:    ${YELLOW}$GRAD_ACCUM${NC}"
echo -e "Effective batch size:     ${YELLOW}$EFFECTIVE_BATCH${NC}"
echo -e "Learning rate:            ${YELLOW}$LR${NC}"
echo -e "Training steps:           ${YELLOW}$STEPS${NC}"
echo -e "Warmup steps:             ${YELLOW}$WARMUP_STEPS${NC}"
echo -e "CPU offload:              ${YELLOW}$([ -n "$CPU_OFFLOAD" ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "Mixed precision (AMP):    ${YELLOW}$([ -n "$AMP_FLAG" ] && echo "Disabled" || echo "Enabled")${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA may not be available.${NC}"
    exit 1
fi

# Check number of available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo -e "${RED}Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available${NC}"
    exit 1
fi

echo -e "${GREEN}Detected $AVAILABLE_GPUS GPUs${NC}"
echo ""

# Build the command
CMD="torchrun --nproc_per_node=$NUM_GPUS multigpu_train.py \
    --model $MODEL \
    --ae_dim $AE_DIM \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --lr $LR \
    --steps $STEPS \
    --warmup_steps $WARMUP_STEPS \
    $CPU_OFFLOAD \
    $AMP_FLAG \
    $EXTRA_ARGS"

echo -e "${BLUE}Launching training...${NC}"
echo -e "${YELLOW}Command: $CMD${NC}"
echo ""

# Launch training
eval $CMD

echo ""
echo -e "${GREEN}Training completed!${NC}"
