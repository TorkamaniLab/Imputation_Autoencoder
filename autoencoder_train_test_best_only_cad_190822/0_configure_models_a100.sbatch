#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=250G
#SBATCH --partition=stsi
#SBATCH --time=600:00:00
#SBATCH --job-name=0_configure_models
#SBATCH --output=%x.oe%j
#SBATCH --error=%x.oe%j
#SBATCH --gres=gpu:a100:4

####load modules
module purge
#module load python/3.8.3
module load pytorch/1.7.1py38-cuda
#module load cuda/10.2
module load samtools/1.10
module load R
#pip3 install cyvcf2 --user


#sbatch --export=input=input.cfg,out_root=chr22 --job-name=0_configure_models 0_configure_models.sbatch
#sbatch --export=input=/mnt/stsi/stsi0/raqueld/imputator/autoencoder_tuning_pipeline/input.cfg,out_root=/mnt/stsi/stsi0/raqueld/imputator/autoencoder_tuning_pipeline/chr22_models --job-name=0_configure_models 0_configure_models.sbatch

cd $SLURM_SUBMIT_DIR


#run $ngpus GPUs in parallel
gsstarttime=$(date +%s)

echo -e "bash 0_configure_models.sh $input $out_root"
bash 0_configure_models.sh $input $out_root

gsendtime=$(date +%s)

gsruntime=$((gsendtime-gsstarttime))


echo "Configuration run time: $gsruntime"

