#!/bin/sh 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=highmem
#SBATCH --time=340:00:00

####PBS -q stsi 
####PBS -l mem=32gb
####PBS -l nodes=1:ppn=16
####PBS -l walltime=128:00:00
####PBS -j oe

#module load python/3.6.3
module load python/3.8.3
module load samtools

    #echo "workdir: $PBS_O_WORKDIR"
    #cd $PBS_O_WORKDIR

    echo "workdir: $SLURM_SUBMIT_DIR"
    cd $SLURM_SUBMIT_DIR

    start=`date +%s`
#    echo running "parallel -j 16 < $cmd"
#    parallel -j 16 < $cmd

for array in $(echo "AFFY6 AXIOM OMNI2M OMNI1M"); do
    for i in {1..22}; do echo "bash minimac.sh $i $array"; bash minimac.sh $i $array; done
done
#for i in {1..22}; do echo -e "bash minimac.sh $i AXIOM; bash run_imputation_jobs_SLURM_${i}_AXIOM.sh"; done > parallel_minimac_AXIOM.sh; parallel -j 22 < parallel_minimac_AXIOM.sh
#for i in {1..22}; do echo -e "bash minimac.sh $i OMNI5M; bash run_imputation_jobs_SLURM_${i}_OMNI5M.sh"; done > parallel_minimac_OMNI5M.sh; parallel -j 22 < parallel_minimac_OMNI5M.sh
#for i in {1..22}; do echo -e "bash minimac.sh $i OMNI2M; bash run_imputation_jobs_SLURM_${i}_OMNI2M.sh"; done > parallel_minimac_OMNI2M.sh; parallel -j 22 < parallel_minimac_OMNI2M.sh

    end=`date +%s`

    runtime=$((end-start))

    echo "Total runtime using 16 cores, 32GB RAM: ${runtime}"
