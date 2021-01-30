if [ -z $1 ] || [ -z $2 ]; then
    echo "usage: bash generate_validation_commands.sh <models_folder> <config_file> <optional_best_only>"
    echo "exemplo: bash generate_validation_commands.sh 1_155032673-155469304 input.cfg"
    exit
fi

inference_script="$PWD/make_inference_commands.sh"
evaluation_script="$PWD/make_evaluation_commands.sh"
selection_script="$PWD/2_pick_best_model_for_full_training.sh"
plot_script="$PWD/plot_evaluation_results_per_variant.R"

model_folder=$1
cfg=$2

cd $model_folder

train_root=$(cat $cfg | grep "TRAIN_DIR" | awk '{print $NF}')
suffix=$(cat BATCH_ID)
train_script=$(cat BATCH_ID)
minimac_suffix=.masked.imputed_minimac.dose.vcf_per_variant_results.txt
VMVpath=$(cat INPUT)
VMV=$(basename $VMVpath)

region=$(echo $VMV | sed -e 's/.*haplotypes\.//g' | sed -e 's/.*_//g')
chr=$(basename $model_folder | cut -f2 -d '_')

for i in $(grep "VAL_GA_DIR" $cfg | tr -d ' '); do 
    idx=$(echo $i | tr '.' ' ' | tr '=' ' ' | cut -f 2 -d ' ')
    val_root=$(echo $i | tr '=' ' ' | cut -f 2 -d ' ')
    cmd0="cat $VMVpath | cut -f 1-5 | grep -v '#' > ${VMV}.1-5"
    val_wgs=$(cat $cfg | tr -d ' ' | tr '=' '\t' | grep -w "^VAL_WGS_DIR\.$idx" | awk '{print $NF}')
    
    if [ -z $3 ]; then
        cmd1="bash $inference_script IMPUTATOR_$VMV ${VMV}.1-5 $val_root inference_output_$idx > run_inference.sh\n\nparallel -j 16 < run_inference.sh"
        cmd2="bash $evaluation_script inference_output_$idx $val_root $val_wgs evaluation_output_$idx > run_evaluation.sh\n\parallel -j 16 < run_evaluation.sh"
        tsv_list="evaluation_output_$idx/*model*.*per_variant*.tsv"
    else
        cmd1="bash $inference_script IMPUTATOR_$VMV ${VMV}.1-5 $val_root inference_output_$idx  | grep \"_F\.\" > run_inference.sh\n\n parallel -j 16 < run_inference.sh"
        cmd2="bash $evaluation_script inference_output_$idx $val_root $val_root evaluation_output_$idx  | grep \"_F\.\" > run_evaluation.sh\n\parallel -j 16 < run_evaluation.sh"
        tsv_list="evaluation_output_$idx/*model*_F.*per_variant*.tsv"
    fi

    VAL=$(basename $(find ${val_root}_minimac4 | grep $region | grep ${minimac_suffix}))
    phased=$(echo ${val_root}_minimac4/$VAL | sed -e 's/_unphased_/_/g')
    custom_files="--custom_files $phased ${val_root}_minimac4/$VAL"
    custom_title=$(basename $model_folder | tr '_' ':' | sed -e 's/^/chr/g')

    if [ -z "$VAL" ]; then
        custom_files=""
        custom_names=""
    else
        custom_names="--custom_names phased_minimac unphased_minimac"
    fi

    if [ -z ${3} ]; then
        cmd3="Rscript $plot_script $tsv_list --threshold -1 $custom_files $custom_names --custom_title $custom_title --out_dir plots_$idx"    
        echo -e "$cmd0\n\n$cmd1\n\n$cmd2\n\n$cmd3\n\n"
    else
        cmd3="Rscript $plot_script $tsv_list --threshold -1 $custom_files $custom_names --custom_title $custom_title --out_dir full_training_plots_$idx"    
        echo -e "$cmd0\n\n$cmd1\n\n$cmd2\n\n$cmd3\n\n"
    fi


done

if [ -z ${3} ]; then
    cmd4="bash $selection_script plots_"
    echo -e "$cmd4"
fi

