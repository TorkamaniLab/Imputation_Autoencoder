#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:4
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=96
##SBATCH --mem=240G
#SBATCH --partition=stsi
#SBATCH --time=600:00:00
#SBATCH --job-name=1_train_GS
#SBATCH --output=%x.oe%j
#SBATCH --error=%x.oe%j

######BSUB -P bif119
######BSUB -W 2:00
######BSUB -nnodes 1
######BSUB -q batch
######BSUB -J test_job
#######BSUB -o logs/job%J.out
#######BSUB -e logs/job%J.log


#sbatch --export=indir=models_22_30236109-30646938 --job-name=1_GS_models_22_30236109-30646938 1_train_GS_a100.sbatch
#while read line; do echo $line; rm GS_models_$line.[oe]*; sbatch --export=indir=models_$line,gpu=3 --job-name=GS_models_$line train_GS_1GPU_a100.sbatch; done < to_run.txt.003;

#WHAT THIS STEP DOES
#1. detect how many gpus are available
#2. calculate how many models can run per GPU based on the VMV size and VRAM limit
#3. parallelize all models across all GPUs in the whole node
#4. verifies if the job was successful or failed and need to rerun, returning either 0 or non-zero status


#insert here modules and commands
#module load open-ce/0.1-0
#conda activate cloned_env
#source ~/.bashrc
#module load bzip2

####load modules
module purge

module load samtools/1.10
module load R
#pip3 install cyvcf2 --user

#V100, TitanV
#module load python/3.8.3
#module load cuda/10.2

#A100
#module load pytorch/1.7.1py38-cuda

#TEST A100
module load pytorch/1.8.0py38-cuda

#debugging
#echo
echo -e "Work dir is $SLURM_SUBMIT_DIR"
#echo -e "Work dir is $LS_SUBCWD"
echo -e "VMV is $indir, GPU a100 $ngpus GPUs"
#echo -e "The shell is $SHELL"
#echo

#module -t list

#cd $LS_SUBCWD
cd $SLURM_SUBMIT_DIR

#A100 test to decrease I/O overhead, try this later
subdir=$(basename $indir)
new_indir=/tmp/$subdir
#clean dir if previous failed job still there!
if [ -d /tmp/$subdir ]; then
    rm -rf /tmp/$subdir
fi
rsync -rtv $indir/* $new_indir/ 
bash 1_train_GS.sh $new_indir A100
rsync -rtv $new_indir/* $indir/
rm -rf $new_indir

#bash 1_train_GS.sh $indir


nmodels=$(ls -l $indir/IMPUTATOR_*/*.pth | grep -v "_F.pth" | wc -l)

if [ $nmodels -lt 90 ]; then
    "Error, less than 90 models ran successfully, please check errors and rerun this job. Exiting with non-zero status."
    exit 1
else
    echo "$nmodels models completed successfully. Job done."
    exit 0
fi
