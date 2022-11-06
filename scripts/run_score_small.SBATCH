#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --partition=cs
#SBATCH --job-name=test_kd1860
#SBATCH --output=test.out
#SBATCH --error=error.err

module purge

singularity exec $nv \
            --overlay /scratch/$USER/my_env/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/miniconda3/bin/activate; 
            python /scratch/$USER/DSGA_1006_capstone/scripts/score_sentences_small.py"