#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --partition=shared
#SBATCH --time=600:00:00
#SBATCH --job-name=2_validate_and_select_best
#SBATCH --output=%x.oe%j
#SBATCH --error=%x.oe%j

######BSUB -P bif119
######BSUB -W 2:00
######BSUB -nnodes 1
######BSUB -q batch
######BSUB -J test_job
#######BSUB -o logs/job%J.out
#######BSUB -e logs/job%J.log


#sbatch --export=mdir=chr22/22_17274081-17382360,input=input.cfg --job-name=4_validate_full_training 4_validate_full_training.sbatch

module load samtools
#module load python/3.6.3
module load python/3.8.3
#pip3 install multiprocessing-logging
#pip3 install argparse
module load R
#R -> install.packages('argparse')


cd $SLURM_SUBMIT_DIR

bash 4_validate_full_training.sbatch $mdir $input
