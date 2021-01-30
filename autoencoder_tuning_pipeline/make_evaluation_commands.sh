script=$(dirname $0)/Compare_imputation_to_WGS.py

imputed_dir=$1

ga_dir=$2

wgs_dir=$3

out_dir=$4

if [ -z $out_dir ]; then

    echo "usage: bash make_inference_commands.sh <imputed_dir> <ga_dir> <wgs_dir> <out_dir>"

    echo "example: bash make_evaluation_commands.sh /raid/pytorch_random_search/inference_output /raid/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_AFFY6 /raid/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged ."
    exit

fi

if [ ! -d $out_dir ]; then
    mkdir -p  $out_dir
fi

for imputed_path in $imputed_dir/*.vcf; do

    #bgzip -c $imputed_path > $imputed_path.gz
    #tabix -p vcf -f $imputed_path.gz

    imputed_name=$(basename $imputed_path)

    ga_name=$(basename $imputed_path | sed -e 's/\.imputed\..*//g')

    ga_path=$(find $ga_dir | grep -w "${ga_name}\.gz$")
    #ga_path=$(find $ga_dir | grep -w "${ga_name}$") #this returns error in cyvcf2


    if [ -z $ga_path ]; then

        echo -e "genotype array file not found in $ga_dir. Searched for $ga_name. Generated search pattern from imputed path $imputed_path"
        exit
    fi

    wgs_name=$(basename $ga_path | sed -e 's/\.masked\..*//g')
    wgs_path=$(find $wgs_dir | grep -w "${wgs_name}\.gz$")

    if [ -z $wgs_path ]; then

        echo -e "wgs file not found in $wgs_dir. Searched for $wgs_name"
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
done


