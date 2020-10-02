
#bash 100_random_hyperparameters.sh make_training_script.sh /raid/chr22/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 4"
commands=$1

if [ -z $2 ] || [ -z $3 ]; then
    echo "usage: bash make_training_script_from_template.sh template.sh input.vcf max_gpus"
    echo "example: bash make_training_script_from_template.sh 100_random_hyperparameters.sh /raid/chr22/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 4"
    exit
fi


VMV_name=$(basename $2)

n_gpus=$3

suffix=$(echo $VMV_name | sed -e 's/.*\.haplotypes//g')

nvar=$(grep -v "#" $2 | wc -l)

nmask=$((nvar-5))
mrate=$(bc -l <<< "$nmask/$nvar" | sed -e 's/^\./0\./g')

bashname="${VMV_name}_${1}"

gpu=0
input=$2

#CUDA_VISIBLE_DEVICES=<my_GPU_id> python3 DSAE_TORCH_ARG.py --input <my_input_file> --min_mask <my_min_mask> 

while read line; do

     echo -e $line | sed -e "s/<my_GPU_id>/$gpu/g" | sed -e "s~<my_input_file>~$input~g" | sed -e "s~<my_min_mask>~0.80~g" | sed -e "s~<my_max_mask>~$mrate~g"

     ((gpu=gpu+1))

     if [ $gpu -eq $n_gpus ]; then
        gpu=0
     fi

done < $1 > $bashname
echo
echo -e "Sequential training script generated at $bashname"
echo
for i in $(seq 0 1 $n_gpus); do
    if [ $i -eq $n_gpus ]; then
        break
    fi
    grep "CUDA_VISIBLE_DEVICES=$i " $bashname > $bashname.GPU$i;
    split -l 1 -a 3 -d $bashname.GPU$i $bashname.GPU${i}.
    for j in $bashname.GPU${i}.[0-9][0-9][0-9]; do echo -e "bash $j 1> $j.out 2> $j.log"; done > $bashname.GPU${i}.parallel.sh
    echo "Parallel training script for GPU $i at $bashname.GPU${i}.parallel.sh"

done
echo
echo "Parallel run automation example, if you want to distribute multiple jobs/models per GPU."
echo "Let's say we are running 3 models per GPU (12 models total if you have 4 GPUs, 6 if you ave 2 GPUs, etc), for example, then the parallel runs would be like:"
echo
for i in $(seq 0 1 $n_gpus); do
    if [ $i -eq $n_gpus ]; then
        break
    fi
    echo "nohup parallel -j 3 < $bashname.GPU${i}.parallel.sh &"
done
echo
echo "You should test different values for parallel's -j argument to find out what is the maximum number of parallel models you will be able to run in a single GPU before reaching the VRAM or CPU bottleneck."
echo "If you are using a cluster with SLURM/TORQUE, put the \"parallel -j 3 < $bashname.<gpu_id>.parallel.sh\" inside a SLURM sbatch job script or TORQUE qsub script"
