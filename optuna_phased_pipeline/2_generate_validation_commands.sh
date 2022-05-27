N_THREADS=2

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
#minimac_suffix=.masked.imputed_minimac.dose.vcf_per_variant_results.txt
competitor_suffix=.masked.imputed_COMPETITOR.dose.vcf_per_variant_results.txt

#NOT WORKING WITH NEW VERSION
#VMVpath=$(cat INPUT)
#VMV=$(basename $VMVpath)

#NOT WORKING ANYMORE BECAUSE OF NEW IMPUTATOR SYNTAX
#region=$(echo $VMV | sed -e 's/.*haplotypes\.//g' | sed -e 's/.*_//g' | sed -e 's/\.gz$//g')
region=$(echo $(basename $1) | tr '_' '\t' | cut -f 2)

chr=$(basename $model_folder | cut -f1 -d '_')

VMVpath=$(find $(grep "^TRAIN_DIR" $cfg | tr -d ' ' | tr '=' ' ' | cut -f 2 -d ' ') | grep $region | grep "\.gz$")

VMV=$(basename $VMVpath)


for i in $(grep "^VAL_GA_DIR" $cfg | tr -d ' '); do 
    idx=$(echo $i | tr '.' ' ' | tr '=' ' ' | cut -f 2 -d ' ')
    #val_root=$(echo $i | tr '=' ' ' | cut -f 2 -d ' ')
    #cmd0="cat $VMVpath | cut -f 1-5 | grep -v '#' > ${VMV}.1-5"
    #val_wgs=$(cat $cfg | tr -d ' ' | tr '=' '\t' | grep -w "^VAL_WGS_DIR\.$idx" | awk '{print $NF}')
    val_root=$(echo $i | tr '=' ' ' | cut -f 2 -d ' ' | sed -e "s/{1\.\.22}/$chr/g")
    cmd0="(zcat $VMVpath || cat $VMVpath) | cut -f 1-5 | grep -v '#' > ${VMV}.1-5"
    val_wgs=$(cat $cfg | tr -d ' ' | tr '=' '\t' | grep -w "^VAL_WGS_DIR\.$idx" | awk '{print $NF}' | sed -e "s/{1\.\.22}/$chr/g")
    
    if [ -z $3 ]; then
        #NOT WORKING IN NEW VERSION
        #cmd1="bash $inference_script IMPUTATOR_$VMV ${VMV}.1-5 $val_root inference_output_$idx > run_inference.sh\n\nparallel -j $N_THREADS < run_inference.sh"
        cmd1="bash $inference_script IMPUTATOR ${VMV}.1-5 $val_root inference_output_$idx > run_inference.sh\n\nparallel -j $N_THREADS < run_inference.sh"
        cmd2="bash $evaluation_script inference_output_$idx $val_root $val_wgs evaluation_output_$idx > run_evaluation.sh\n\nparallel -j $N_THREADS < run_evaluation.sh"
        tsv_list="evaluation_output_$idx/*model*.*per_variant*.tsv"
    else
        #NOT WORKING IN NEW VERSION
        #cmd1="bash $inference_script IMPUTATOR_$VMV ${VMV}.1-5 $val_root inference_output_$idx  | grep \"_F\.\" > run_inference.sh\n\nparallel -j $N_THREADS < run_inference.sh"
        cmd1="bash $inference_script IMPUTATOR ${VMV}.1-5 $val_root inference_output_$idx  | grep \"_F\.\" > run_inference.sh\n\nparallel -j $N_THREADS < run_inference.sh"
        cmd2="bash $evaluation_script inference_output_$idx $val_root $val_wgs evaluation_output_$idx  | grep \"_F\.\" > run_evaluation.sh\n\nparallel -j $N_THREADS < run_evaluation.sh"
        tsv_list="evaluation_output_$idx/*model*_F.*per_variant*.tsv"
    fi

    #minimac
    #VAL=$(basename $(find ${val_root}_minimac4 | grep $region | grep ${minimac_suffix}))
    #phased=$(echo ${val_root}_minimac4/$VAL | sed -e 's/_unphased_/_/g')
    #custom_files="--custom_files $phased ${val_root}_minimac4/$VAL"
    #if [ -z "$VAL" ]; then
    #    custom_files=""
    #    custom_names=""
    #else
    #    custom_names="--custom_names phased_minimac unphased_minimac"
    #fi

    custom_files="--custom_files"
    custom_names="--custom_names"
    #generalized to all competitors
    my_error=0
    for competitor in minimac4 beagle5 impute5; do
        my_name=$(echo $competitor | sed -e 's/[0-9]//g')
        if [ $competitor = minimac4 ]; then
            my_suffix=$(echo $competitor_suffix | sed -e "s/COMPETITOR/${my_name}/g")
        else
            my_suffix=$(echo $competitor_suffix | sed -e "s/COMPETITOR\.dose/${my_name}/g")
        fi
        VAL=$(basename $(find ${val_root}_${competitor} | grep $region | grep ${my_suffix}))
        phased=$(echo ${val_root}_$competitor/$VAL | sed -e 's/_unphased_/_/g')
        if [ ! -z "$VAL" ]; then
            custom_files="$custom_files $phased"
            custom_names="$custom_names phased_$my_name"
        else
            my_error=1
            echo "ERROR $competitor NOT FOUND. TRIED: find ${val_root}_${competitor} | grep $region | grep ${my_suffix}"
        fi
    done
    if [ $my_error -eq 1 ]; then
        custom_files=""
        custom_names=""
    fi

    custom_title=$(basename $model_folder | tr '_' ':' | sed -e 's/^/chr/g')

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

