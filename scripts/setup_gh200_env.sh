#!/bin/bash
# Environment setup script for DINOv2 training on Grace Hopper (GH200) nodes.
# Source this script before launching training:
#   source scripts/setup_gh200_env.sh

# --- Micromamba environment ---
module load micromamba 2>/dev/null || true
micromamba activate /gpfs/data/pearson-lab/PERSONAL/saideep/.conda/envs/dinov2-gh200

# --- NCCL tuning for NVLink-C2C ---
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# --- CUDA 12 memory allocator ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Enable TF32 for matmuls (H100) ---
export NVIDIA_TF32_OVERRIDE=1
