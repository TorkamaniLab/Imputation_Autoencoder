#sbatch --export=train_list=/mnt/stsi/stsi0/raqueld/imputator/autoencoder_tuning_pipeline/chr22_models/full_training_list.txt --job-name=3_train_best_model_a100 3_train_best_model_a100.sbatch

#WHAT THIS STEP DOES
#1. check how much VRAM this model needs
#2. check the VRAM still available in GPUi
#3. there is room in the VRAM for this model?
#4. if yes, submit the model to GPUi
#5. if no, i=i+i
#6. if none of all GPUs is available, wait and try again later

train_list=$1

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
else
    echo "Not using A100, nor V100 flags ${2}"
fi


while read train_script; do

    result=-1
    echo $train_script
    curr_dir=$(pwd)
    train_dir=$(dirname $train_script)
    nvar=$(cat $train_dir/NVAR)
    batchi=$(cat $train_dir/BATCH_ID)
    ngpus=$(nvidia-smi --query-gpu=gpu_name --format=csv | tail -n +2 | wc -l)

    #1. check how much VRAM this model needs, previously calculated by 1_train_GS_a100.sbatch
    #mem_needed=$(python3 tools/estimate_VRAM_needed_for_autoencoder.py $nvar | tail -n 1 | awk '{print $NF}' | sed -e 's/MiB//g')
    mem_needed=$(cat $train_dir/VRAM)

    while [ $result -eq -1 ]; do
        result=-1
        echo "Submitting ${train_script}"

    
        #2. check the VRAM still available in GPUs
        gpui_mem=$(nvidia-smi --query-gpu=gpu_name,memory.free --format=csv | tail -n +2 | awk '{print NR-1 "\t" $(NF-1)}' | sort -k2g -k1gr | tail -n 1)
        gpui=$(echo $gpui_mem | awk '{print $1}')
        mem_avail=$(echo $gpui_mem | awk '{print $2}')
        echo "GPU available: $gpui, memory available: $mem_avail, memory needed: $mem_needed"
        #3. there is room in the VRAM for this model?
        if [ $mem_avail -gt $mem_needed ]; then
            #4. if yes, submit the model to GPUi
            cd $train_dir
            #first time
            sed -i -e "s/<my_GPU_id>/$gpui/g" $batchi.best
            #retraining failed job
            sed -i -e "s/CUDA_VISIBLE_DEVICES=[0-9] /CUDA_VISIBLE_DEVICES=$gpui /g" $batchi.best
            ${GPU_flags[${gpui}]}bash $train_script &
            echo -e "${GPU_flags[${gpui}]}bash $train_script"
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
            echo "Submited ${train_script}, PID $result"
            echo "Train script content:"
            cat $train_dir/$batchi.best
            echo "Directory: $train_dir"
            echo -e "$result" >> $pids
            #allow some time to the job allocate the whole VRAM it needs, then go to the next
            sleep 180
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

