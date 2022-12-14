#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --array=0-4
#SBATCH --cpus-per-task=2
#SBATCH --time=5:00:00
#SBATCH --mem=64GB
#SBATCH --partition=cs
#SBATCH --job-name=cluster
#SBATCH --output=../dataset/outputs/cluster/shard_%A_%a.out
#SBATCH --error=../dataset/outputs/cluster/shard_%A_%a.err

module purge

singularity exec $nv \
            --overlay /scratch/$USER/my_env/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/miniconda3/bin/activate; 
            python /scratch/$USER/DSGA_1006_capstone/scripts/cluster.py $SLURM_ARRAY_TASK_ID"