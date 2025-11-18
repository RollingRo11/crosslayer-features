# Multi-GPU Crosscoder Training

This document explains how to use the multi-GPU version of the crosscoder trainer with Fully Sharded Data Parallel (FSDP), mixed precision training, and gradient accumulation.

## Overview

The `multigpucrosscoder.py` file provides a distributed training implementation that:

- **Fully Sharded Data Parallel (FSDP)**: Shards model parameters, gradients, and optimizer states across multiple GPUs for maximum memory efficiency
- **Automatic Mixed Precision (AMP)**: Uses FP16 for gradient reduction to speed up training and reduce memory usage
- **Gradient Accumulation**: Simulates larger batch sizes by accumulating gradients over multiple steps
- **Distributed Data Loading**: Each GPU processes different data samples with synchronized normalization
- **Coordinated Checkpointing**: Saves full model checkpoints from rank 0 only

## Quick Start

### Single Node, 8 GPUs

```bash
torchrun --nproc_per_node=8 multigpu_train.py \
    --model gemma2-2b \
    --ae_dim 32768 \
    --batch_size 256 \
    --gradient_accumulation_steps 4 \
    --steps 50000 \
    --lr 5e-5
```

### Single Node, 4 GPUs with Gradient Accumulation

```bash
torchrun --nproc_per_node=4 multigpu_train.py \
    --model gemma2-2b \
    --ae_dim 32768 \
    --batch_size 512 \
    --gradient_accumulation_steps 8 \
    --steps 50000 \
    --lr 5e-5
```

### With CPU Offloading (for very large models)

```bash
torchrun --nproc_per_node=8 multigpu_train.py \
    --model gemma2-2b \
    --ae_dim 65536 \
    --batch_size 256 \
    --fsdp_cpu_offload \
    --gradient_accumulation_steps 4 \
    --steps 50000
```

## Configuration Parameters

### Basic Parameters (same as single-GPU)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `gpt2` | Model to train on: `gpt2` or `gemma2-2b` |
| `--ae_dim` | `32768` | Autoencoder dimension (sparse feature space) |
| `--batch_size` | `2048` | **Batch size per GPU** (not total) |
| `--lr` | `5e-5` | Learning rate |
| `--steps` | `50000` | Total training steps |
| `--l1_coeff` | `0.8` | L1 sparsity coefficient |
| `--warmup_steps` | `5000` | Learning rate warmup steps |

### Multi-GPU Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gradient_accumulation_steps` | `1` | Number of steps to accumulate gradients before updating |
| `--use_amp` | `True` | Enable automatic mixed precision (FP16) |
| `--no_amp` | - | Disable automatic mixed precision |
| `--fsdp_cpu_offload` | `False` | Offload parameters to CPU (saves GPU memory) |
| `--fsdp_min_num_params` | `1e6` | Minimum parameters for FSDP wrapping |

## Understanding Effective Batch Size

The **effective batch size** is calculated as:

```
Effective Batch Size = batch_size × gradient_accumulation_steps × num_gpus
```

**Examples:**

1. **8 GPUs, batch_size=256, accumulation=4**
   - Effective batch size = 256 × 4 × 8 = 8,192

2. **4 GPUs, batch_size=512, accumulation=8**
   - Effective batch size = 512 × 8 × 4 = 16,384

3. **2 GPUs, batch_size=1024, accumulation=2**
   - Effective batch size = 1024 × 2 × 2 = 4,096

## Memory Optimization Strategies

### If you run out of memory:

1. **Enable CPU offloading**:
   ```bash
   --fsdp_cpu_offload
   ```

2. **Reduce batch size per GPU**:
   ```bash
   --batch_size 128  # Instead of 256
   ```

3. **Increase gradient accumulation** (to maintain effective batch size):
   ```bash
   --gradient_accumulation_steps 8  # Instead of 4
   ```

4. **Reduce autoencoder dimension** (if acceptable):
   ```bash
   --ae_dim 16384  # Instead of 32768
   ```

## Multi-Node Training

For training across multiple nodes, you'll need to set up the distributed environment properly:

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    multigpu_train.py \
    --model gemma2-2b \
    --ae_dim 32768 \
    --batch_size 256

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    multigpu_train.py \
    --model gemma2-2b \
    --ae_dim 32768 \
    --batch_size 256
```

## Implementation Details

### FSDP Strategy

The implementation uses `ShardingStrategy.FULL_SHARD`, which:
- Shards model parameters across all GPUs
- Shards gradients across all GPUs
- Shards optimizer states across all GPUs
- Maximizes memory efficiency

### Mixed Precision Configuration

When `--use_amp` is enabled (default):
- Parameters: FP32 (full precision)
- Gradient reduction: FP16 (half precision)
- Buffers: FP32 (full precision)

This provides speed benefits while maintaining training stability.

### Gradient Accumulation

The training loop accumulates gradients over multiple mini-batches:
1. Zero gradients once at the start
2. Forward pass on mini-batch
3. Backward pass with scaled loss
4. Repeat steps 2-3 for `gradient_accumulation_steps`
5. Clip gradients and update parameters

### Distributed Data Loading

Each GPU:
- Gets a different subset of data from the Pile dataset
- Uses rank-specific random shuffling
- Shares normalization statistics via `all_reduce`
- Ensures diverse data across all ranks

### Checkpointing

Only rank 0 saves checkpoints:
- FSDP gathers full model state to rank 0
- Checkpoint includes model, optimizer, and config
- All ranks synchronize with barriers before/after saving

## Monitoring Training

### WandB Logging

Only the main process (rank 0) logs to Weights & Biases:
- Metrics are aggregated across all GPUs before logging
- Project name: `crosscoder-multigpu`
- Run name format: `{model}_ae{ae_dim}_gpu{world_size}`

### Console Output

Only rank 0 prints training progress:
```
Step 100/50000 | Loss: 2.3451 | Recon: 1.8234 | ... | L0: 125.3 | Dead Features: 234
```

### Metrics Tracked

All metrics are averaged across GPUs:
- `losses/loss`: Total loss
- `losses/recon_loss`: Reconstruction loss
- `losses/sparsity_loss`: L1 sparsity loss
- `losses/preact_loss`: Preactivation loss
- `stats/l0_sparsity`: Average active features per sample
- `stats/dead_features`: Number of never-activated features
- `hyperparams/lr`: Current learning rate
- `hyperparams/l1_coeff`: Current L1 coefficient
- `stats/W_dec_norm`: Decoder weight norm

## Troubleshooting

### NCCL Timeout Errors

If you see NCCL timeout errors, try:
```bash
export NCCL_TIMEOUT=3600  # Increase timeout to 1 hour
export NCCL_DEBUG=INFO    # Enable debug logging
```

### OOM (Out of Memory) Errors

1. Check GPU memory usage:
   ```bash
   nvidia-smi
   ```

2. Try the memory optimization strategies above

3. Consider using CPU offloading:
   ```bash
   --fsdp_cpu_offload
   ```

### Gradient Synchronization Issues

If gradients aren't synchronizing properly:
- Ensure all GPUs are on the same network
- Check that NCCL is properly installed
- Verify that the CUDA version matches PyTorch's CUDA version

### Different Results than Single-GPU

This is expected due to:
- Different random seeds per rank
- Different data ordering
- Different numerical precision (FP16 reduction)

Results should be statistically similar but not identical.

## Performance Tips

1. **Use the maximum batch size** that fits in GPU memory
2. **Enable AMP** for faster training (enabled by default)
3. **Use gradient accumulation** to simulate larger batches
4. **Monitor GPU utilization** with `nvidia-smi` or `nvitop`
5. **Ensure fast storage** for dataset streaming
6. **Use InfiniBand** for multi-node training if available

## Comparison with Single-GPU

### Advantages of Multi-GPU:
- ✅ Faster training (near-linear speedup with num GPUs)
- ✅ Larger effective batch sizes
- ✅ Can train larger models (with FSDP sharding)
- ✅ More data diversity per step

### Disadvantages:
- ❌ More complex setup and debugging
- ❌ Requires multiple GPUs
- ❌ Slightly different numerical results
- ❌ More communication overhead

## Example Training Commands

### GPT-2 on 4 GPUs (Fast training)
```bash
torchrun --nproc_per_node=4 multigpu_train.py \
    --model gpt2 \
    --ae_dim 32768 \
    --batch_size 512 \
    --gradient_accumulation_steps 2 \
    --steps 30000 \
    --lr 5e-5 \
    --warmup_steps 3000
```

### Gemma-2-2B on 8 GPUs (Production training)
```bash
torchrun --nproc_per_node=8 multigpu_train.py \
    --model gemma2-2b \
    --ae_dim 65536 \
    --batch_size 256 \
    --gradient_accumulation_steps 4 \
    --steps 100000 \
    --lr 5e-5 \
    --warmup_steps 5000 \
    --log_interval 50 \
    --save_interval 5000
```

### Memory-Constrained Training (with CPU offload)
```bash
torchrun --nproc_per_node=8 multigpu_train.py \
    --model gemma2-2b \
    --ae_dim 131072 \
    --batch_size 128 \
    --fsdp_cpu_offload \
    --gradient_accumulation_steps 8 \
    --steps 100000 \
    --lr 5e-5
```

## Architecture Differences from Single-GPU

The multi-GPU implementation maintains the same model architecture as the single-GPU version but adds:

1. **Distributed initialization** via `setup_distributed()`
2. **FSDP model wrapping** for parameter sharding
3. **Gradient scaler** for mixed precision training
4. **Synchronized buffer refresh** with shared normalization stats
5. **Metric aggregation** across all GPUs via `all_reduce`
6. **Distributed checkpointing** with full state dict gathering

The core crosscoder model (encoder, decoder, JumpReLU) remains unchanged, ensuring compatibility with single-GPU checkpoints (though you may need to unwrap FSDP state dicts).

## Further Reading

- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
