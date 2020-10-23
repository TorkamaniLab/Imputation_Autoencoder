script=Compare_imputation_to_WGS.py


if [ -z $4 ]; then
    echo
    echo "usage: bash make_inference_commands.sh <imputed_dir> <ga_dir> <wgs_dir> <out_dir>"
    echo
    echo "example: bash make_evaluation_commands.sh /raid/pytorch_random_search/inference_output /raid/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_AFFY6 /raid/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged ."
    echo
    echo "<imputed_dir> directory where the inference results were generated (this script supports results from make_inference_commands.sh only!)."
    echo "<ga_dir> directory were the genotype array or \"masked\" input files are (files used as input by the inference function."
    echo "<wgs_dir> directory where the ground truth WGS files are."
    echo "<out_dir> directory where to save the evaluation results."
    echo
    exit

fi

imputed_dir=$(readlink -f $1)

ga_dir=$(readlink -f $2)

wgs_dir=$(readlink -f $3)

out_dir=$(readlink -f $4)


if [ ! -d $out_dir ]; then
    mkdir -p  $out_dir
fi


if [ ! -f ${out_dir}/${script}  ]; then
    cp $script $out_dir/
fi



for imputed_path in $imputed_dir/*.vcf; do

    #bgzip -c $imputed_path > $imputed_path.gz
    #tabix -p vcf -f $imputed_path.gz

    imputed_name=$(basename $imputed_path)

    ga_name=$(basename $imputed_path | sed -e 's/\.imputed\..*//g')

    ga_path=$(find $ga_dir | grep -w "${ga_name}\.gz$")


    if [ -z $ga_path ]; then

        echo -e "genotype array file not found in $ga_dir. Searched for $ga_name"
        exit
    fi

    wgs_name=$(basename $ga_path | sed -e 's/\.masked\..*//g')
    wgs_path=$(find $wgs_dir | grep -w "${wgs_name}\.gz$")

    if [ -z $wgs_path ]; then

        echo -e "genotype array file not found in $wgs_dir. Searched for $wgs_name"
        exit
    fi

    sout=$out_dir/${imputed_name}_per_sample.tsv
    vout=$out_dir/${imputed_name}_per_variant.tsv

    #echo $imputed_path
    #echo $ga_path
    #echo $wgs_path
    #echo $sout
    #echo $vout

    cmd="bgzip -c $imputed_path > $imputed_path.gz; tabix -p vcf -f $imputed_path.gz; python3 $script --wgs $wgs_path --imputed $imputed_path.gz --ga $ga_path --sout $sout --vout $vout"

    echo $cmd
done > $out_dir/run_evaluation.sh


echo "Evaluation script generated at ${out_dir}/run_evaluation.sh"
echo "To run evaluations sequentially:"
echo "cd ${out_dir}; bash run_evaluation.sh"
echo "To run inferences in parallel:"
echo "cd ${out_dir}; parallel -j 4 < run_evaluation.sh"


