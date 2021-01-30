script=$(dirname $0)/inference_function.py

model_dir=$1

pos=$2

ga_dir=$3

out_dir=$4

if [ -z $out_dir ]; then

    echo "usage: bash make_inference_commands.sh <model_dir> <pos_file.1-5> <ga_dir> <out_dir>"
    echo "example: bash make_inference_commands.sh /raid/pytorch_random_search/models/IMPUTATOR_HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 /raid/chr22/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1.1-5 /raid/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_AFFY6 ./"
    exit

fi

if [ ! -d $out_dir ]; then
    mkdir -p  $out_dir
fi

region=$(basename $pos | sed -e 's/.*haplotypes\.//g' | sed -e 's/.*_//g' | sed -e 's/\..*//g')
chr=$(basename $pos | sed -e 's/.*\.chr/chr/g' | sed -e 's/\..*//g')
#echo $region
#echo $chr
#echo -e "find $ga_dir | grep -w $region | grep "masked.gz$\|masked$""
ga_path=$(find $ga_dir | grep $region | grep "masked.gz$\|masked$" | head -n 1)

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


    cmd="python3 $script $pos $ga_path $model_dir --model_name $model_name --output $out_path"
    #cmd="python3 $script $pos $ga_path $model_dir --model_name $model_name --output $out_path --use_gpu"
    echo $cmd
done
