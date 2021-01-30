#sbatch --export=indir=models_22_30236109-30646938 --job-name=1_GS_models_22_30236109-30646938 1_train_GS_a100.sbatch
#while read line; do echo $line; rm GS_models_$line.[oe]*; sbatch --export=indir=models_$line,gpu=3 --job-name=GS_models_$line train_GS_1GPU_a100.sbatch; done < to_run.txt.003;

indir=$1


#WHAT THIS STEP DOES
#1. detect how many gpus are available
#2. calculate how many models can run per GPU based on the VMV size and VRAM limit
#3. parallelize all models across all GPUs in the whole node
#4. verifies if the job was successful or failed and need to rerun, returning either 0 or non-zero status



ngpus=$(nvidia-smi --query-gpu=gpu_name --format=csv | tail -n +2 | wc -l)


nvar=$(cat $indir/NVAR)

#identify max VRAM used per model and estimate how many models can run in parallel
mem_per_proc=$(python3 tools/estimate_VRAM_needed_for_autoencoder.py $nvar | tail -n 1 | awk '{print $NF}' | sed -e 's/MiB//g')
echo -e "$mem_per_proc" > $indir/VRAM

max_mem_avail=$(nvidia-smi --query-gpu=gpu_name,memory.total --format=csv | tail -n 1 | awk '{print $(NF-1)}')
npar=$(echo "$max_mem_avail/$mem_per_proc" | bc)

re='^[0-9]+$'
if ! [[ $mem_per_proc =~ $re ]] ; then
    echo -e "error: estimate_VRAM_needed_for_autoencoder.py failed, the models probably don't fit in VRAM, check error log, max VRAM available: $max_mem_avail"; exit 1
elif [ $npar -lt 1 ]; then
    echo -e "error: estimate_VRAM_needed_for_autoencoder.py failed, the models probably don't fit in VRAM, check error log, max VRAM available: $max_mem_avail"; exit 1
elif [ $npar -gt 6 ]; then
    echo -e "Maximum theoretical number of processes per GPU is $npar, but it will be reduced to 6 to avoid I/O overhangs and RAM/CPU bottleneck"
    npar=6
fi

echo -e "mem_per_proc = $mem_per_proc; max_mem_avail = $max_mem_avail; npar = $npar"

echo -e "cd $indir;"
cd $indir; 

batchi=$(cat BATCH_ID)
input=$(cat INPUT)

echo -e "batchi = $batchi; nvar = $nvar; input = $input"

#split data across $ngpus GPUs
nhp=$(cat $batchi | wc -l)
chunksize=$(echo -e "$nhp/$ngpus" | bc)
split -l $chunksize -d -a 1 $batchi $batchi.GPU
mgpu=$((ngpus-1))

echo -e "nhp = $nhp; chunksize = $chunksize; mgpu = $mgpu"

if [ "${2}" = "A100" ]; then
    declare -a GPU_flags
    GPU_flags+=("numactl --physcpubind=0-23 -l ")
    GPU_flags+=("numactl --physcpubind=48-71 -l ")
    GPU_flags+=("numactl --physcpubind=24-47 -l ")
    GPU_flags+=("numactl --physcpubind=72-95 -l ")
    echo -e "A100 flags ${GPU_flags[@]}"
elif [ "${2}" = "V100" ]; then
    declare -a GPU_flags
    GPU_flags+=("numactl --physcpubind=0-23 -l ")
    echo -e "V100 flags ${GPU_flags[@]}"
else
    echo "Not using A100, nor V100 flags ${2}"
fi

for i in $(seq 0 1 $mgpu); do
    sed -i -e "s/<my_GPU_id>/$i/g" $batchi.GPU$i
    sed -i -e "s/CUDA_VISIBLE_DEVICES=[0-9]/CUDA_VISIBLE_DEVICES=$i/g" $batchi.GPU$i

    echo -e "${GPU_flags[${i}]}parallel -j $npar < $batchi.GPU$i"
done > run.sh

#run $ngpus GPUs in parallel
gsstarttime=$(date +%s)

echo -e "parallel -j $ngpus < run.sh"
parallel -j $ngpus < run.sh

gsendtime=$(date +%s)

gsruntime=$((gsendtime-gsstarttime))


echo "GS run time: $gsruntime"

nmodels=$(ls -l IMPUTATOR_*/*.pth | wc -l)
if [ $nmodels -lt 90 ]; then
    "Error, less than 90 models ran successfully, please check errors and rerun this job. Exiting with non-zero status."
    exit 1
else
    echo "$nmodels models completed successfully. Job done."
    exit 0
fi
