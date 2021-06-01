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

for i in {1..22}; do echo -e "bash run_unphasing_SLURM.sh chr${i}_AFFY6"; done > parallel_AFFY6.sh; parallel -j 16 < parallel_AFFY6.sh
for i in {1..22}; do echo -e "bash run_unphasing_SLURM.sh chr${i}_AXIOM"; done > parallel_AXIOM.sh; parallel -j 16 < parallel_AXIOM.sh
#for i in {1..22}; do echo -e "bash run_unphasing_SLURM.sh chr${i}_OMNI5M"; done > parallel_OMNI5M.sh; parallel -j 16 < parallel_OMNI5M.sh
for i in {1..22}; do echo -e "bash run_unphasing_SLURM.sh chr${i}_OMNI2M"; done > parallel_OMNI2M.sh; parallel -j 16 < parallel_OMNI2M.sh
for i in {1..22}; do echo -e "bash run_unphasing_SLURM.sh chr${i}_OMNI1M"; done > parallel_OMNI1M.sh; parallel -j 16 < parallel_OMNI1M.sh

    end=`date +%s`

    runtime=$((end-start))

    echo "Total runtime using 16 cores, 32GB RAM: ${runtime}"
