#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --partition=stsi
#SBATCH --time=600:00:00
#SBATCH --job-name=1_train_GS
#SBATCH --output=%x.oe%j
#SBATCH --error=%x.oe%j
#SBATCH --gres=gpu:a100:4

######BSUB -P bif119
######BSUB -W 2:00
######BSUB -nnodes 1
######BSUB -q batch
######BSUB -J test_job
#######BSUB -o logs/job%J.out
#######BSUB -e logs/job%J.log


#sbatch --export=train_list=/mnt/stsi/stsi0/raqueld/imputator/autoencoder_tuning_pipeline/chr22_models/full_training_list.txt --job-name=3_train_best_model_a100 3_train_best_model_a100.sbatch
#sbatch --export=train_list=/mnt/stsi/stsi0/raqueld/imputator/xgb_best_models_chr22_unphased_autoencoder/best_hyperparameters_from_500000_xgb_a100.sh.003-006 --job-name=3_train_best_model_a100 3_train_custom_best_model_a100.sbatch
#sbatch --export=train_list=/mnt/stsi/stsi0/raqueld/imputator/xgb_best_models_chr22_unphased_autoencoder/xgb_best_models_a100.sh.023,out=cad_190822_models --job-name=3_train_best_model_a100_xgb023 3_train_custom_best_model_a100.sbatch

#WHAT THIS STEP DOES
#1. check how much VRAM this model needs
#2. check the VRAM still available in GPUi
#3. there is room in the VRAM for this model?
#4. if yes, submit the model to GPUi
#5. if no, i=i+i
#6. if none of all GPUs is available, wait and try again later

#insert here modules and commands
#module load open-ce/0.1-0
#conda activate cloned_env
#source ~/.bashrc
#module load bzip2

####load modules
module purge
#module load python/3.8.3
module load pytorch/1.7.1py38-cuda
#module load cuda/10.2
module load samtools/1.10
module load R
#pip3 install cyvcf2 --user



#debugging
#echo
echo -e "Work dir is $SLURM_SUBMIT_DIR"
#echo -e "Work dir is $LS_SUBCWD"
echo -e "Train list is $train_list, GPU a100"
#echo -e "The shell is $SHELL"
#echo

#module -t list

#cd $LS_SUBCWD
cd $SLURM_SUBMIT_DIR

#bash 3_train_best_model.sh $train_list A100
#bash 3_train_best_model_DGX.sh $train_list A100
bash 3_train_custom_best_model.sh $train_list A100 $out
