#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --partition=shared
#SBATCH --time=600:00:00
#SBATCH --job-name=0_configure_models
#SBATCH --output=%x.oe%j
#SBATCH --error=%x.oe%j

######BSUB -P bif119
######BSUB -W 2:00
######BSUB -nnodes 1
######BSUB -q batch
######BSUB -J test_job
#######BSUB -o logs/job%J.out
#######BSUB -e logs/job%J.log


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

