#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --partition=stsi
#SBATCH --time=600:00:00
#SBATCH --job-name=train
#SBATCH --output=%x.oe%j
#SBATCH --error=%x.oe%j
#SBATCH --gres=gpu:a100:1


#for i in {0..3}; do
#    echo -e "sbatch --export=cmd=train_best.GPU00$i.sh --job-name=train_best.GPU00$i train_bestM_1GPU_3models_a100_from_cmd.sbatch"
#    sbatch --export=cmd=train_best.GPU00$i.sh --job-name=train_best.GPU00$i train_bestM_1GPU_3models_a100_from_cmd.sbatch
#done

####load modules
module purge
module load pytorch/1.8.0py38-cuda
#module load python/3.8.3
#module load pytorch/1.7.1py38-cuda
#module load cuda/10.2
module load samtools/1.10
module load R
#pip3 install cyvcf2 --user

#debugging
#echo
echo -e "Work dir is $SLURM_SUBMIT_DIR"
echo -e "python is "
which python
echo -e "python3 is "
which python3
#module -t list

cd $SLURM_SUBMIT_DIR


echo -e "running parallel -j 3 < $cmd"
parallel -j 3 < $cmd
