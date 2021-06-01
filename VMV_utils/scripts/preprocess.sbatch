#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=highmem
#SBATCH --time=340:00:00
#SBATCH --job-name="validation"

####PBS -q stsi
####PBS -l mem=32gb
####PBS -l nodes=1:ppn=16
####PBS -l walltime=128:00:00
####PBS -j oe

#module load python/3.6.3
module load python/3.8.3
module load samtools
module load java #beagle


#echo "workdir: $PBS_O_WORKDIR"
#cd $PBS_O_WORKDIR

echo "workdir: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

start=`date +%s`
echo running "parallel -j 16 < $cmd"
parallel -j 16 < $cmd
end=`date +%s`

runtime=$((end-start))

echo "Total runtime using 16 cores, 32GB RAM: ${runtime}"

