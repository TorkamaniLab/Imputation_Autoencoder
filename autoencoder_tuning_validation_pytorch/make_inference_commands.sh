script=inference_function.py
log=configure_logging.py
util=genotype_conversions.py
train=DSAE_TORCH_ARG.py

model_dir=$(readlink -f $1)

pos=$(readlink -f $2)

ga_dir=$(readlink -f $3)

out_dir=$(readlink -f $4)

if [ -z $out_dir ]; then

    echo "usage: bash make_inference_commands.sh <model_dir> <pos_file.1-5> <ga_dir> <out_dir>"
    echo "example: bash make_inference_commands.sh /raid/pytorch_random_search/models/IMPUTATOR_HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 /raid/chr22/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1.1-5 /raid/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_AFFY6 ./inference_output"
    exit

fi


if [ ! -d $out_dir ]; then
    mkdir -p  $out_dir
fi

if [ ! -f ${out_dir}/${script}  ]; then
    cp $script $out_dir/
    cp $log $out_dir/
    cp $util $out_dir/
    cp $train $out_dir/
fi

region=$(basename $pos | sed -e 's/.*haplotypes\.//g' | sed -e 's/\..*//g')
chr=$(basename $pos | sed -e 's/.*\.chr/chr/g' | sed -e 's/\..*//g')
#echo $region
#echo $chr

ga_path=$(find $ga_dir | grep -w $region | grep "masked$")
#echo $ga_path

if [ ! -f $ga_path ]; then
    echo -e "genotype array file not found in $ga_dir. Searched for region $region, suffix masked"

    exit
fi

ga_name=$(basename $ga_path)


for model_path in ${model_dir}/*.pth; do

    #param_path=$(echo ${model_path} | sed -e 's/pth$/_param\.py/g')
    #param_name=$(basename ${param_path} | sed -e 's/\.py$//g')
    model_name=$(basename ${model_path} | sed -e 's/\.pth$//g')
    out_path="$out_dir/${ga_name}.imputed.${model_name}.vcf"


    cmd="python3 $script $pos $ga_path $model_dir --model_name $model_name --output $out_path --use_gpu"
    echo $cmd

done > ${out_dir}/run_inference.sh

echo "Inference script generated at ${out_dir}/run_inference.sh"
echo "To run inferences sequentially:"
echo "cd ${out_dir}; bash run_inference.sh"
echo "To run inferences in parallel:"
echo "cd ${out_dir}; parallel -j 4 < run_inference.sh"
