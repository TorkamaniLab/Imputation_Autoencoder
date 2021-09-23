#sbatch --export=indir=models_22_30236109-30646938 --job-name=1_GS_models_22_30236109-30646938 1_train_GS_a100.sbatch
#while read line; do echo $line; rm GS_models_$line.[oe]*; sbatch --export=indir=models_$line,gpu=3 --job-name=GS_models_$line train_GS_1GPU_a100.sbatch; done < to_run.txt.003;

indir=$1


#WHAT THIS STEP DOES
#1. detect how many gpus are available
#2. calculate how many models can run per GPU based on the VMV size and VRAM limit
#3. parallelize all models across all GPUs in the whole node
#4. verifies if the job was successful or failed and need to rerun, returning either 0 or non-zero status

#1. check how much VRAM this model needs
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
fi

echo -e "mem_per_proc = $mem_per_proc; max_mem_avail = $max_mem_avail; npar = $npar"

echo -e "cd $indir;"
cd $indir;

batchi=$(cat BATCH_ID)
input=$(cat INPUT)
mem_needed=$(cat VRAM)
pids=${batchi}.PID_list.txt
if [ -f $pids ]; then
    rm $pids
fi


echo -e "batchi = $batchi; nvar = $nvar; input = $input"

#split data across $ngpus GPUs
nhp=$(cat $batchi | wc -l)

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


gsstarttime=$(date +%s)

while read train_script; do
    result=-1
    echo $train_script
    input_file=$(echo $train_script | sed -e 's/.*--input //g' | sed -e 's/ .*//g')
    model_id=$(echo $train_script | sed -e 's/.*--model_id //g' | sed -e 's/ .*//g')
    echo $train_script > train_${model_id}.sh

    while [ $result -eq -1 ]; do
        result=-1
        echo "Submitting ${train_script}"
        echo -e "Input file: $input_file"

        #2. check the VRAM still available in GPUs
        gpui_mem=$(nvidia-smi --query-gpu=gpu_name,memory.free --format=csv | tail -n +2 | awk '{print NR-1 "\t" $(NF-1)}' | sort -k2g -k1gr | tail -n 1)
        gpui=$(echo $gpui_mem | awk '{print $1}')
        mem_avail=$(echo $gpui_mem | awk '{print $2}')
        echo "GPU available: $gpui, memory available: $mem_avail, memory needed: $mem_needed"
        #3. there is room in the VRAM for this model?
        if [ $mem_avail -gt $mem_needed ]; then
            #4. if yes, submit the model to GPUi
            #first time
            sed -i -e "s/<my_GPU_id>/$gpui/g" train_${model_id}.sh
            #retraining failed job
            sed -i -e "s/CUDA_VISIBLE_DEVICES=[0-9] /CUDA_VISIBLE_DEVICES=$gpui /g" train_${model_id}.sh
            ${GPU_flags[${gpui}]}bash train_${model_id}.sh &
            echo -e "${GPU_flags[${gpui}]}bash train_${model_id}.sh"
            PID=$!
            result=$PID
        fi
        #no GPUs have VRAM available, wait 10 minutes
        #5. if none of all GPUs is available, wait and try again later
        if [ $result -eq -1 ]; then
            sleep 600
        fi


        if [ $result -ne -1 ]; then
            echo "Submited $indir/train_${model_id}.sh, PID $result"
            echo "Script content:"
            cat train_${model_id}.sh
            echo "Directory: $indir"
            echo -e "$result" >> $pids
            #allow some time to the job allocate the whole VRAM it needs, then go to the next
            sleep 120
        fi
    done
done < $batchi

while true; do
    if [ -s $pids ] ; then
        for pid in $(cat $pids); do
            echo "Checking pids $pid"
            kill -0 "$pid" 2>/dev/null || sed -i "/^$pid$/d" $pids
        done
    else
        echo "All process completed"
        break
    fi
    sleep 180;
done

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
