#sbatch --export=train_list=/mnt/stsi/stsi0/raqueld/imputator/autoencoder_tuning_pipeline/chr22_models/full_training_list.txt --job-name=3_train_best_model_a100 3_train_best_model_a100.sbatch

#WHAT THIS STEP DOES
#1. check how much VRAM this model needs
#2. check the VRAM still available in GPUi
#3. there is room in the VRAM for this model?
#4. if yes, submit the model to GPUi
#5. if no, i=i+i
#6. if none of all GPUs is available, wait and try again later

train_list=$1

mroot=$3
if [ -z $3 ]; then
    mroot=chr22_models
fi

gsstarttime=$(date +%s)

pids=${train_list}.PID_list.txt

if [ -f $pids ]; then
    rm $pids
fi


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
elif [ "${2}" = "summit" ]; then
    declare -a GPU_flags
    GPU_flags+=("numactl --physcpubind=0-27 -l ")
    GPU_flags+=("numactl --physcpubind=28-55 -l ")
    GPU_flags+=("numactl --physcpubind=56-83 -l ")
    GPU_flags+=("numactl --physcpubind=88-115 -l ")
    GPU_flags+=("numactl --physcpubind=116-143 -l ")
    GPU_flags+=("numactl --physcpubind=144-171 -l ")
    echo -e "summit cluster (ORNL) flags ${GPU_flags[@]}"
else
    echo "Not using A100, nor V100 flags ${2}"
fi


while read train_script; do
    result=-1
    echo $train_script
    input_file=$(echo $train_script | sed -e 's/.*--input //g' | sed -e 's/ .*//g')
    #mem_needed=$(python3 tools/estimate_VRAM_needed_for_autoencoder.py $nvar | tail -n 1 | awk '{print $NF}' | sed -e 's/MiB//g')
    region=$(echo $input_file | sed -e 's/.*\.haplotypes\.//' | sed -e 's/.*_//g' | sed -e 's/\..*//g')
    chr=$(echo $input_file | sed -e 's/.*chr//g' | sed -e 's/\..*//g')
    mdir=$mroot/${chr}_${region}
    cp DSAE_TORCH_ARG.py $mdir/
    mem_needed=$(cat $mdir/VRAM)
    curr_dir=$(pwd)
    model_id=$(echo $train_script | sed -e 's/.*--model_id //g' | sed -e 's/ .*//g')
    echo $train_script > $mdir/train_best_${model_id}.sh

    while [ $result -eq -1 ]; do
        result=-1
        echo "Submitting ${train_script}"

        echo -e "Input file: $input_file"
        nvar=$(grep -v "#" $input_file | wc -l)
        #1. check how much VRAM this model needs, previously calculated by 1_train_GS_a100.sbatch
        ngpus=$(nvidia-smi --query-gpu=gpu_name --format=csv | tail -n +2 | wc -l)

        #2. check the VRAM still available in GPUs
        gpui_mem=$(nvidia-smi --query-gpu=gpu_name,memory.free --format=csv | tail -n +2 | awk '{print NR-1 "\t" $(NF-1)}' | sort -k2g -k1gr | tail -n 1)
        gpui=$(echo $gpui_mem | awk '{print $1}')
        mem_avail=$(echo $gpui_mem | awk '{print $2}')
        echo "GPU available: $gpui, memory available: $mem_avail, memory needed: $mem_needed"
        #3. there is room in the VRAM for this model?
        if [ $mem_avail -gt $mem_needed ]; then
            #4. if yes, submit the model to GPUi
            cd $mdir
            #first time
            sed -i -e "s/<my_GPU_id>/$gpui/g" train_best_${model_id}.sh
            #retraining failed job
            sed -i -e "s/CUDA_VISIBLE_DEVICES=[0-9] /CUDA_VISIBLE_DEVICES=$gpui /g" train_best_${model_id}.sh
            ${GPU_flags[${gpui}]}bash train_best_${model_id}.sh 1>> train_best.${model_id}.out 2> train_best_${model_id}.log &
            echo -e "${GPU_flags[${gpui}]}bash train_best_${model_id}.sh 1>> train_best.${model_id}.out 2> train_best_${model_id}.log"
            PID=$!
            result=$PID
            cd $curr_dir
        fi
        #5. if no, wait, try again later

        #no GPUs have VRAM available, wait 10 minutes
        #6. if none of all GPUs is available, wait and try again later
        if [ $result -eq -1 ]; then
            sleep 600
        fi




        if [ $result -ne -1 ]; then
            echo "Submited $mdir/train_best_${model_id}.sh, PID $result"
            echo "Script content:"
            cat $mdir/train_best_${model_id}.sh
            echo "Directory: $mdir"
            echo -e "$result" >> $pids
            #allow some time to the job allocate the whole VRAM it needs, then go to the next
            n=0
            while [ $n -lt 18 ]; do
                # echo ${n}
                if grep -q "Execution\|Cancelling" ${mdir}/$(less $mdir/train_best_${model_id}.sh | tr ' ' '\n' | tail -3 | head -1); then
                    grep "Execution\|Cancelling" ${mdir}/$(less $mdir/train_best_${model_id}.sh | tr ' ' '\n' | tail -3 | head -1)
                    break
                else
                    sleep 10
                    let n=n+1
                fi
            done
            # sleep 180
        fi

    done
done < $train_list

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


echo "Full training run time: $gsruntime"

