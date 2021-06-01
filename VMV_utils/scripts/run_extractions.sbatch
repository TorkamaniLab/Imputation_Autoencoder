#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=highmem
#SBATCH --time=96:00:00



#######!/bin/sh 
########PBS -q stsi 
########PBS -l mem=16gb
#########PBS -l nodes=1:ppn=1
#########PBS -l walltime=96:00:00
#########PBS -j oe

#To run this script at command line:
#for i in {1..22}; do 
#    qsub -v chr=${i},outdir=AFR,infile=/mnt/stsi/stsi0/raqueld/minimac_imputation_results/ARIC/5_phase/c1_ARIC_WGS_Freeze3/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-5.chr${i}.phased.vcf.gz run_extractions.qsub
#done
#for i in {1..22}; do 
#    sbatch --export chr=${i},outdir=AFR,infile=/mnt/stsi/stsi0/raqueld/minimac_imputation_results/ARIC/5_phase/c1_ARIC_WGS_Freeze3/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-5.chr${i}.phased.vcf.gz run_extractions.sbatch
#done


#chr chromosome
#a ancestry code, 1 for EUR, 5 for AFR
#outdir output directory where to save results

cd $SLURM_SUBMIT_DIR
#####cd $PBS_O_WORKDIR

bash extract_blocks_from_HRC_regions.sh $chr $infile $outdir
