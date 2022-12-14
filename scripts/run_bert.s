#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --time=5:00:00
#SBATCH --mem=64GB
#SBATCH --partition=cs
#SBATCH --job-name=bert
#SBATCH --output=../dataset/output/bert/shard_%A_%a.out
#SBATCH --error=../dataset/output/bert/shard_%A_%a.err

module purge

singularity exec $nv \
            --overlay /scratch/$USER/my_env/overlay-15GB-500K.ext3:ro \
            /scratch/wz2247/singularity/images/pytorch_22.08-py3.sif  \
            /bin/bash -c "source /ext3/miniconda3/bin/activate;
            python /scratch/$USER/DSGA_1006_capstone/scripts/bert.py $SLURM_ARRAY_TASK_ID"