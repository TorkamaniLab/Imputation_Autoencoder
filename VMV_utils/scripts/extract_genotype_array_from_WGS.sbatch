#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=highmem
#SBATCH --time=340:00:00



#####!/bin/sh 
#######PBS -q stsi 
#######PBS -l mem=16gb
#######PBS -l nodes=1:ppn=1
######PBS -l walltime=96:00:00
#########PBS -j oe

#To run this script at command line: `$ for i in {1..22}; do qsub -v chr=${i} extract_FHS_OMNI2M_from_WGS.qsub; done`
#To run this script at command line: `$ for i in {1..22}; do sbatch --export chr=${i} extract_FHS_OMNI2M_from_WGS.sbatch; done`

#    cd $PBS_O_WORKDIR
echo "workdir: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

    bash extract_genotype_array_from_WGS.sh ${chr} ${rfile} ${array}
    echo "Done making ${chr} ${rfile} ${array} folder!"
