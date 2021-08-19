#/mnt/stsi/stsi0/raqueld/VMV_VCF_Extractions/chr22/

#bash configure_models.sh input.cfg out_root

full_models_per_node=40
hp=xgb_best_models.sh
train_script=DSAE_TORCH_ARG.py
#inference_script=inference_function.py
#accuracy_script=Compare_imputation_to_WGS.py
#inf_dependency1=configure_logging.py
#inf_dependency2=genotype_conversions.py
#plot_script=plot_evaluation_results_per_variant.R

if [ -z $1 ]; then
    echo "Please provide config file."
    echo "Usage: bash configure_models.sh input.cfg /path/to/output/root_dir"
    exit
fi


if [ -z $out_root ]; then
    out_root=$PWD
elif [ ! -d $out_root ]; then
    mkdir -p $out_root
    abs=$(readlink -e $2)
    out_root=$abs
fi

echo "Model training and validation directories will be at $out_root"

echo "#model directories" > $out_root/model_dir_list.txt

method=$(cat $1 | grep -v "#" | grep "^METHOD" | tr '=' ' ' | awk '{print $NF}')
traindir=$(cat $1 | grep -v "#" | grep "^TRAIN_DIR" | tr '=' ' ' | awk '{print $NF}')
echo $traindir

wdir=$PWD

vram_script=$wdir/tools/estimate_VRAM_needed_for_autoencoder.py

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

    hp_name=$(basename $hp)
    
    cp $hp $mdir/
    cp $train_script $mdir/

    cd $mdir

    echo "$nvar" > NVAR
    input=$i
    echo "$input" > INPUT

    #identify max VRAM used per model and estimate how many models can run in parallel
    mem_per_proc=$(python3 $vram_script $nvar | tail -n 1 | awk '{print $NF}' | sed -e 's/MiB//g')
    echo -e "$mem_per_proc" > VRAM


    if [ $method = "RANDOM" ]; then

        bashname="${VMV_name}_${hp_name}"
        while read line; do
            mi=$(echo $line | sed -e 's/.*--model_id //g' | sed -e 's/ .*//g')
            echo -e $line | sed -e "s~<my_input_file>~$input~g" | sed -e "s~<my_min_mask>~0.80~g" | sed -e "s~<my_max_mask>~$mrate~g" | sed -e "s~<my_input_file>~$input~g" | sed -e "s~<my_min_mask>~0.80~g" | sed -e "s~<my_max_mask>~$mrate~g" | sed -e "s~$~ 1\> $bashname\.$mi\.out 2\> $bashname\.$mi\.log~g"
        done < $hp_name > $bashname
        split -l 100 -a 3 -d $bashname ${bashname}.

        #start at batch 0, or 1
        echo "$bashname.000" > BATCH_ID
                
    elif [ $method = "RAYTUNE" ]; then
        #insert RAYTUNE CONFIGURATION HERE
        echo -e "RAYTUNE COMMING SOON!!!"
        exit
    else
        echo -e "method not supported: $method. Please revise the value of METHOD in your config file $1"
        exit
    fi


    cd $wdir
        
    
#    break
done

cat ${out_root}/*/*.000 > ${out_root}/full_training_list.txt
split -l $full_models_per_node -a 3 -d ${out_root}/full_training_list.txt ${out_root}/full_training_list.txt
