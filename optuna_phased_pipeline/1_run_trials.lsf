#!/bin/bash
##### project id must be bif138 (our official project ID, all lower case)
#BSUB -P bif138
#####24 hours
#BSUB -W 24:00
#BSUB -nnodes 9
######## 6 gpus per node
#########BSUB -alloc_flags gpumps
#BSUB -q batch-hm
#BSUB -o logs/job_%J.out
#BSUB -e logs/job_%J.log


#bsub -env "all, input=chr22_models/22_17274081-17382360/22_17274081-17382360_train.sh.001" 1_run_trials_test.lsf
#/gpfs/alpine/bif138/scratch/raqueld/Imputation_Autoencoder/optuna_phased_pipeline/chr22_models/22_17274081-17382360/22_17274081-17382360_train.sh.001

source /ccs/proj/bif138/env.sh

#debugging
echo -e "Work dir is $LS_SUBCWD"

gsstarttime=$(date +%s)


indir=$(dirname $input)
inscript=$(basename $input)
subdir=$LS_SUBCWD

echo -e "cd $indir && bash $inscript"
echo -e "Running $inscript, check $input.* for progress info and log messages."

cd $indir 

#run 50 scripts of 10 trials each
$subdir/with_postgresql.sh 22_17274081-17382360 jsrun -n 54 -g 1 -c 7 -b packed:7 --latency_priority gpu-cpu,cpu-cpu bash $inscript

gsendtime=$(date +%s)

gsruntime=$((gsendtime-gsstarttime))


echo "Run time: $gsruntime"


echo "Done. Exiting..."


exit

