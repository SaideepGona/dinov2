#!/usr/bin/env python3
"""Generate SLURM batch scripts for DINOv2 hyperparameter search.

Produces one SLURM submission script per hyperparameter combination,
plus a master script to submit them all.

Usage:
    python scripts/generate_hp_search.py \
        --base-config configs/my_config.yaml \
        --output-dir /path/to/hp_search_runs \
        --batch-sizes 100 200 400 \
        --learning-rates 1e-4 2e-4 4e-4 \
        --partition ghq \
        --reservation pearson-gpu

    # Then submit all:
    bash /path/to/hp_search_runs/submit_all.sh
"""

import argparse
import itertools
import os
import stat


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{reservation_line}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={output_dir}/slurm_%j.log
#SBATCH --error={output_dir}/slurm_%j.err
{exclude_line}

# --- Environment setup ---
module load micromamba 2>/dev/null || true
micromamba activate {conda_env}

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME={nccl_socket_ifname}
export NCCL_IB_HCA={nccl_ib_hca}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1
export CUDA_VISIBLE_DEVICES=$SLURM_STEP_GPUS

echo "========================================="
echo "Hyperparameter search: {job_name}"
echo "  batch_size={batch_size}"
echo "  base_lr={base_lr}"
echo "  num_workers={num_workers}"
echo "  output_dir={output_dir}"
echo "  node: $(hostname)"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================="

torchrun \\
    --standalone \\
    --nproc_per_node={gpus} \\
    {dinov2_dir}/dinov2/train/train.py \\
    --config-file {base_config} \\
    --output-dir {output_dir} \\
    train.batch_size_per_gpu={batch_size} \\
    train.num_workers={num_workers} \\
    optim.base_lr={base_lr} \\
    {extra_overrides}
"""


def generate_scripts(args):
    os.makedirs(args.output_dir, exist_ok=True)

    combos = list(itertools.product(args.batch_sizes, args.learning_rates))
    print(f"Generating {len(combos)} runs:")

    submit_lines = ["#!/bin/bash", f"# Submit all {len(combos)} HP search jobs", ""]
    summary_lines = []

    for batch_size, lr in combos:
        run_name = f"bs{batch_size}_lr{lr}"
        run_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        job_name = f"dv2_{run_name}"

        reservation_line = (
            f"#SBATCH --reservation={args.reservation}"
            if args.reservation else ""
        )
        exclude_line = (
            f"#SBATCH --exclude={args.exclude}"
            if args.exclude else ""
        )

        extra_overrides = " \\\n    ".join(args.extra_overrides) if args.extra_overrides else ""

        script_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            partition=args.partition,
            reservation_line=reservation_line,
            gpus=args.gpus,
            cpus=args.cpus,
            mem=args.mem,
            time=args.time,
            output_dir=run_dir,
            exclude_line=exclude_line,
            conda_env=args.conda_env,
            batch_size=batch_size,
            base_lr=lr,
            num_workers=args.num_workers,
            dinov2_dir=args.dinov2_dir,
            base_config=args.base_config,
            nccl_socket_ifname=args.nccl_socket_ifname,
            nccl_ib_hca=args.nccl_ib_hca,
            extra_overrides=extra_overrides,
        )

        script_path = os.path.join(run_dir, "submit.sh")
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)

        submit_lines.append(f'echo "Submitting {run_name}..."')
        submit_lines.append(f"sbatch {script_path}")
        submit_lines.append("")

        # Effective LR after sqrt scaling
        import math
        eff_lr = lr * math.sqrt(batch_size / 1024.0)
        summary_lines.append(
            f"  {run_name:30s}  batch_size={batch_size:4d}  "
            f"base_lr={lr}  eff_lr={eff_lr:.6f}"
        )
        print(f"  {run_name}: batch_size={batch_size}, base_lr={lr}, eff_lr={eff_lr:.6f}")

    # Write submit_all.sh
    submit_all_path = os.path.join(args.output_dir, "submit_all.sh")
    with open(submit_all_path, "w") as f:
        f.write("\n".join(submit_lines) + "\n")
    os.chmod(submit_all_path, os.stat(submit_all_path).st_mode | stat.S_IEXEC)

    # Write summary
    summary_path = os.path.join(args.output_dir, "runs_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Hyperparameter search: {len(combos)} runs\n")
        f.write(f"Base config: {args.base_config}\n")
        f.write(f"LR scaling: sqrt_wrt_1024 (eff_lr = base_lr * sqrt(batch_size/1024))\n\n")
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nGenerated {len(combos)} scripts in {args.output_dir}")
    print(f"Submit all: bash {submit_all_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for DINOv2 hyperparameter search"
    )

    # Required
    parser.add_argument("--base-config", required=True,
                        help="Path to base YAML config file")
    parser.add_argument("--output-dir", required=True,
                        help="Root directory for all HP search runs")

    # Hyperparameters to search
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[100, 200, 400],
                        help="Batch sizes to search (default: 100 200 400)")
    parser.add_argument("--learning-rates", nargs="+", type=float,
                        default=[1e-4, 2e-4, 4e-4],
                        help="Base learning rates to search (default: 1e-4 2e-4 4e-4)")

    # SLURM settings
    parser.add_argument("--partition", default="ghq",
                        help="SLURM partition (default: ghq)")
    parser.add_argument("--reservation", default=None,
                        help="SLURM reservation (e.g. pearson-gpu)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="GPUs per job (default: 1)")
    parser.add_argument("--cpus", type=int, default=60,
                        help="CPUs per task (default: 60)")
    parser.add_argument("--mem", default="400G",
                        help="Memory per job (default: 400G)")
    parser.add_argument("--time", default="24:00:00",
                        help="Wall time per job (default: 24:00:00)")
    parser.add_argument("--exclude", default=None,
                        help="Nodes to exclude (e.g. cri22cn408)")

    # NCCL settings
    parser.add_argument("--nccl-socket-ifname", default="eno1",
                        help="Network interface for NCCL (default: eno1)")
    parser.add_argument("--nccl-ib-hca", default="mlx5",
                        help="InfiniBand HCA for NCCL (default: mlx5)")

    # Training settings
    parser.add_argument("--num-workers", type=int, default=50,
                        help="DataLoader workers (default: 50)")
    parser.add_argument("--conda-env",
                        default="/gpfs/data/pearson-lab/PERSONAL/saideep/.conda/envs/dinov2-gh200",
                        help="Conda/micromamba environment path")
    parser.add_argument("--dinov2-dir",
                        default="/gpfs/data/pearson-lab/PERSONAL/saideep/projects/UCH_CHEVRIER_WHOLEMOUSE/scripts/mouse_foundation_model_training/dinov2",
                        help="Path to DINOv2 repo")

    # Extra config overrides
    parser.add_argument("--extra-overrides", nargs="*", default=[],
                        help="Additional config overrides (e.g. optim.epochs=200 train.prefetch_factor=32)")

    args = parser.parse_args()
    generate_scripts(args)


if __name__ == "__main__":
    main()
