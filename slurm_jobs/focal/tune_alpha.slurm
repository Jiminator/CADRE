#!/bin/bash
#SBATCH --mem=64G
#SBATCH --output="CADRE.out"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS, 64 requests whole node
#SBATCH --partition=gpuA100x4    # <- one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcrn-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=CADRE
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -t 00:10:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

echo "job is starting on `hostname`"

# run the container binary with arguments: python3 <program.py>
# --bind /projects/bbXX  # add to apptainer arguments to mount directory inside container
cd /scratch/bdkz/jshong/CADRE
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate cadre
# printf "alpha = 0.0\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.0
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.0
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.0

# printf "alpha = 0.1\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.1
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.1
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.1

# printf "alpha = 0.2\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.2
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.2
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.2

# printf "alpha = 0.3\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.3
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.3
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.3

# printf "alpha = 0.4\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.4
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.4
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.4

# printf "alpha = 0.5\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.5
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.5
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.5

# printf "alpha = 0.6\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.6
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.6
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.6

# printf "alpha = 0.7\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.7
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.7
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.7

# printf "alpha = 0.8\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.8
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.8
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.8

# printf "alpha = 0.9\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.9
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.9
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=0.9

# printf "alpha = 1.0\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=1.0
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=1.0
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=1.0

# printf "alpha = auto\n"
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=-1
# time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=-1
time python3 run_cf.py --repository gdsc --model_label cntx-attn-gdsc --focal=True --alpha=-1