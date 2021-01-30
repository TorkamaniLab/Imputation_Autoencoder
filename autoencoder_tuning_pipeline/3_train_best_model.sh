#sbatch --export=train_list=/mnt/stsi/stsi0/raqueld/imputator/autoencoder_tuning_pipeline/chr22_models/full_training_list.txt --job-name=3_train_best_model_a100 3_train_best_model_a100.sbatch

#WHAT THIS STEP DOES
#1. check how much VRAM this model needs
#2. check the VRAM still available in GPUi
#3. there is room in the VRAM for this model?
#4. if yes, submit the model to GPUi
#5. if no, i=i+i
#6. if none of all GPUs is available, wait and try again later

train_list=$1

submit_training () 
    train_script=$(basename $1)
    train_dir=$(dirname $1)
    nvar=$(cat $train_dir/NVAR)
    #1. check how much VRAM this model needs, previously calculated by 1_train_GS_a100.sbatch
    mem_needed=$(cat $train_dir/VRAM)

    ngpus=$(nvidia-smi --query-gpu=gpu_name --format=csv | tail -n +2 | wc -l)
    
    for i in $(seq 1 1 $ngpus); do
        #2. check the VRAM still available in GPUi
        mem_avail=$(nvidia-smi --query-gpu=gpu_name,memory.free --format=csv | tail -n $i | head -n 1 | cut -f 2 -d ' ')
        #3. there is room in the VRAM for this model?
        if [ $mem_avail -gt $mem_needed ]; then
            #4. if yes, submit the model to GPUi
            cd $train_dir
            bash $train_script &
            PID=$!
            return $PID
        fi
        #5. if no, i=i+i
    done
    #no GPUs have VRAM available, wait 10 minutes
    #6. if none of all GPUs is available, wait and try again later
    sleep 600
    return -1
}


gsstarttime=$(date +%s)


while read train_script; do
    result=-1
    while [ $result -eq -1 ]; do
        result=-1
        result=$(submit_training $train_script)
        if [ $result -ne -1 ];
            echo -e "$result" >> ${train_list}.PID_list.txt
            #allow some time to the job allocate the whole VRAM it needs, then go to the next
            sleep 180
        fi
    done
done < $train_list

pids=${train_list}.PID_list.txt
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
    sleep 60;
done

gsendtime=$(date +%s)

gsruntime=$((gsendtime-gsstarttime))


echo "Full training run time: $gsruntime"

