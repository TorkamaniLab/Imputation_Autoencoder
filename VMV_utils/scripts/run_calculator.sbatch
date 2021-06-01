#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=shared
#SBATCH --time=340:00:00


######!/bin/sh 
#######PBS -q stsi 
######PBS -l mem=32gb
######PBS -l nodes=1:ppn=16
######PBS -l walltime=128:00:00
#######PBS -j oe
#################PBS -m ea 


#SCRIPT FOR RUNNING MINIMAC IMPUTATION FOR ARIC CHR22
    #cd $PBS_O_WORKDIR
    cd $SLURM_SUBMIT_DIR

    module load samtools
    module load python/3.8.3

    start=`date +%s`
    echo running $cmd
    parallel -j 16 < $cmd

    end=`date +%s`

    runtime=$((end-start))

    echo "Total runtime using 16 cores, 32GB RAM: ${runtime}"
