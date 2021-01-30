if [ -z $1 ]; then
    echo "usage: bash generate_validation_commands_best_model_only.sh <models_folder>"
    echo "exemplo: bash generate_validation_commands_best_model_only.sh models_1_155032673-155469304"
    exit
fi

suffix=_100_random_hyperparameters_500e.sh
#val_root=/mnt/10TB/raqueld/ARIC/EUR_AFR
val_root=/mnt/stsi/stsi0/raqueld/ARIC_VMV/EUR_AFR
minimac_suffix=.masked.imputed_minimac.dose.vcf_per_variant_results.txt
#train_root=/mnt/10TB/raqueld/HRC
train_root=/mnt/stsi/stsi0/raqueld/VMV_VCF_Extractions

for model_folder in $1; do

    #bash make_inference_commands.sh $model_folder 
    train_script=$(find $model_folder -name *$suffix)
    VMV=$(basename $train_script | sed -e "s/${suffix}//g")
    region=$(echo $VMV | sed -e 's/.*haplotypes\.//g' | sed -e 's/.*_//g' | sed -e 's/\..*//g')
    chr=$(echo $model_folder | cut -f2 -d '_')
    cmd0="cat $train_root/chr$chr/${VMV} | cut -f 1-5 | grep -v '#' > $train_root/chr$chr/${VMV}.1-5"
    cmd1="bash make_inference_commands.sh $model_folder/IMPUTATOR_$VMV $train_root/chr$chr/${VMV}.1-5 $val_root/chr${chr}_masked $model_folder/inference_output | grep \"_F\.\"  > $model_folder/run_inference.sh\n\nbash $model_folder/run_inference.sh"
    cmd2="bash make_evaluation_commands.sh $model_folder/inference_output $val_root/chr${chr}_masked $val_root/chr${chr} $model_folder/evaluation_output | grep \"_F\.\" > $model_folder/run_evaluation.sh\n\nbash $model_folder/run_evaluation.sh"
    tsv_list="$model_folder/evaluation_output/*model*_F.*per_variant*.tsv"
    #echo $(find $val_root/chr${chr}_masked_minimac4 -name *$region*${minimac_suffix})
    VAL=$(basename $(find $val_root/chr${chr}_masked_minimac4 | grep $region | grep ${minimac_suffix}))
    #echo VAL $VAL
    custom_files="--custom_files $val_root/chr${chr}_masked_minimac4/$VAL $val_root/chr${chr}_masked_unphased_minimac4/$VAL"
    custom_title=$(echo $model_folder | sed -e 's/models_//g' | tr '_' ':' | sed -e 's/^/chr/g')
    cmd3="Rscript plot_evaluation_results_per_variant.R $tsv_list --threshold -1 $custom_files --custom_names phased_minimac unphased_minimac --custom_title $custom_title --out_dir $model_folder/plots_best"
    cmd4="bash pick_best_model_for_full_training.sh $model_folder/plots/overall_results_per_model.tsv $train_script"
    
    echo -e "$cmd0\n\n$cmd1\n\n$cmd2\n\n$cmd3"

done
