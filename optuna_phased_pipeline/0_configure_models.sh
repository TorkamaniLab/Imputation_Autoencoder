#/mnt/stsi/stsi0/raqueld/VMV_VCF_Extractions/chr22/

#bash configure_models.sh input.cfg out_root

train_script=DSAE_TORCH_ARG_PHASED.py
tool_script1=genotype_conversions_dip_oh.py
tool_script2=custom_samplers.py
#inference_script=inference_function.py
#accuracy_script=Compare_imputation_to_WGS.py
#inf_dependency1=configure_logging.py
#plot_script=plot_evaluation_results_per_variant.R

if [ -z $1 ]; then
    echo "Please provide config file."
    echo "Usage: bash configure_models.sh input.cfg /path/to/output/root_dir"
    exit
fi


if [ -z $2 ]; then
    out_root=$PWD
else
    out_root=$2
fi

if [ ! -d $out_root ]; then
    mkdir -p $out_root
fi
out_root=$(readlink -e $2)

cfg=$(readlink -e $1)

echo "Model training and validation directories will be at $out_root"

echo "#model directories" > $out_root/model_dir_list.txt

traindir=$(cat $cfg | grep -v "#" | grep "^TRAIN_DIR" | tr '=' ' ' | awk '{print $NF}')

sampler=$(cat $cfg | grep -v "#" | grep -w "^sampler" | tr '=' ' ' | awk '{print $NF}')
sampling_res=$(cat $cfg | grep -v "#" | grep -w "^sampling_res" | tr '=' ' ' | awk '{print $NF}')
n_trials=$(cat $cfg | grep -v "#" | grep -w "^n_trials" | tr '=' ' ' | awk '{print $NF}')
trials_per_job=$(cat $cfg | grep -v "#" | grep -w "^trials_per_job" | tr '=' ' ' | awk '{print $NF}')
pruning=$(cat $cfg | grep -v "#" | grep -w "^pruning" | tr '=' ' ' | awk '{print $NF}')
patience=$(cat $cfg | grep -v "#" | grep -w "^patience" | tr '=' ' ' | awk '{print $NF}')
max_models_per_gpu=$(cat $cfg | grep -v "#" | grep -w "^max_models_per_gpu" | tr '=' ' ' | awk '{print $NF}')

mysql=$(cat $cfg | grep -v "#" | grep -w "^mysql" | wc -l)
if [ $mysql -gt 0 ]; then
    flag="--mysql"
    value=$(cat $cfg | grep -v "#" | tr -d ' ' | grep -w "^mysql" | tr '=' ' ' | awk '{print $NF}')
    mysql="$flag $value"
else
    mysql=""
fi

echo $traindir

wdir=$PWD

#remove remaining job list from previous run
if [ -f ${out_root}/full_training_list.txt ]; then
    rm ${out_root}/full_training_list.txt*
fi

for i in $traindir/*.VMV1.gz; do

    region=$(echo $i | sed -e 's/.*\.haplotypes\.//' | sed -e 's/.*_//g' | sed -e 's/\..*//g')
    chr=$(echo $i | sed -e 's/.*chr//g' | sed -e 's/\..*//g')
    echo -e "${chr}_${region}\t$i"
    mdir="${out_root}/${chr}_${region}"
    echo "$mdir" >> $out_root/model_dir_list.txt
    if [ ! -d $mdir ]; then
        mkdir -p $mdir
    fi

    VMV_name=$(basename $i)

    suffix=$(echo $VMV_name | sed -e 's/.*\.haplotypes//g')

    nvar=$(zgrep -v "#" $i | wc -l)

    nmask=$((nvar-5))
    mrate=$(bc -l <<< "$nmask/$nvar" | sed -e 's/^\./0\./g')

    cp $train_script $mdir/
    cp $tool_script1 $mdir/
    cp $tool_script2 $mdir/
    #cp $accuracy_script $mdir/
    #cp $inf_dependency1 $mdir/

    cd $mdir

    echo "$nvar" > NVAR
    input=$i
    echo "$input" > INPUT


    n_jobs=$((n_trials/trials_per_job))
    study_name="${chr}_${region}"
    ga_list=""

    for val in $(grep "^VAL_GA_DIR" $cfg | tr -d ' '); do
        idx=$(echo $val | tr '.' ' ' | tr '=' ' ' | cut -f 2 -d ' ')
        ga_dir=$(echo $val | tr '=' '\t' | awk '{print $NF}' | sed -e "s/{1\.\.22}/$chr/g")
        wgs_dir=$(cat $cfg | tr '=' '\t' | grep -w "^VAL_WGS_DIR\.$idx" | awk '{print $NF}' | sed -e "s/{1\.\.22}/$chr/g")
        ga_path=$(find $ga_dir | grep $region | grep "masked.gz$\|masked$" | head -n 1)
        ga_list="$ga_list --val_input $ga_path"
        if [ $idx -eq 3 ]; then
            break
        fi
    done

    wgs_list=""
    wgs_name=$(basename $ga_path | sed -e 's/\.masked\..*//g')
    wgs_path=$(find $wgs_dir | grep -w "${wgs_name}\.gz$")
    wgs_list="$wgs_list --val_true $wgs_path"

    bashname=${study_name}_train.sh
    for job in $(seq -f "%03g" 1 1 $n_jobs); do
        echo "CUDA_LAUNCH_BLOCKING=1 python3 $train_script --min_mask 0.80 --max_mask $mrate \
        --study_name $study_name --n_trials $trials_per_job --sampler $sampler --patience $patience \
        --sampling_res $sampling_res --pruning $pruning --max_models_per_gpu $max_models_per_gpu \
        --resume 1 --input $input $ga_list $wgs_list $mysql \
        1> $bashname.$job.out 2> $bashname.$job.log"
    done | sed -e 's/        //g' | sed -e 's/  / /g' > $bashname

    if [ ! -z $value ]; then
        python3 $wdir/clean_database.py $value $study_name
    fi

    split -l 1 -a 3 --numeric-suffixes=1 $bashname ${bashname}.
    cd $wdir

#    break
done

for i in ${out_root}/*/*_train.sh.*; do
    echo $i
done > ${out_root}/full_training_list.txt
