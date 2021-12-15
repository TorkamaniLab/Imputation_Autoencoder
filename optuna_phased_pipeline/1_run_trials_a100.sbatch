#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=350G
#SBATCH --partition=stsi
#SBATCH --time=600:00:00
#SBATCH --job-name=0_configure_models
#SBATCH --output=%x.oe%j
#SBATCH --error=%x.oe%j
#SBATCH --gres=gpu:a100:4

####load modules
module purge
#module load python/3.8.3
#module load pytorch/1.7.1py38-cuda
module load pytorch/1.9.0py38-cuda

#module load cuda/10.2
module load samtools/1.10
module load R

echo "using orca from $(which orca)"

#input example:/raid/raqueld/optuna_phased_pipeline/chr22_models/22_38708556-38866010/22_38708556-38866010_train.sh.001
#sbatch --export=input=$input --job-name=1_run_trials 1_run_trials_a100.sbatch
#out=/raid/raqueld/optuna_phased_pipeline/chr22_models/
#while read script; do sbatch --export=input=$script --job-name=1_run_trials 1_run_trials_a100.sbatch; done < $out/full_training_list.txt;


cd $SLURM_SUBMIT_DIR

#declare -a GPU_flags
#GPU_flags+=("numactl --physcpubind=0-23 -l ")
#GPU_flags+=("numactl --physcpubind=48-71 -l ")
#GPU_flags+=("numactl --physcpubind=24-47 -l ")
#GPU_flags+=("numactl --physcpubind=72-95 -l ")
#echo -e "A100 flags ${GPU_flags[@]}"

#${GPU_flags[${2}]}bash train_model_gpu2.sh


#run $ngpus GPUs in parallel
gsstarttime=$(date +%s)


indir=$(dirname $input)
inscript=$(basename $input)

echo -e "cd $indir && bash $inscript"
echo -e "Running $inscript, check $input.* for progress info and log messages."
cd $indir && bash $inscript

gsendtime=$(date +%s)

gsruntime=$((gsendtime-gsstarttime))


echo "Run time: $gsruntime"


