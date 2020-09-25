
#bash 100_random_hyperparameters.sh make_training_script.sh /raid/chr22/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 4"
commands=$1

if [ -z $2 ] || [ -z $3 ]; then
    echo "usage: bash make_training_script_from_template.sh template.sh input.vcf max_gpus"
    echo "example: bash make_training_script_from_template.sh 100_random_hyperparameters.sh examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 4"
    exit
fi


VMV_name=$(basename $2)

n_gpus=$3

suffix=$(echo $VMV_name | sed -e 's/.*\.haplotypes//g')

nvar=$(grep -v "#" $2 | wc -l)

nmask=$((nvar-5))
mrate=$(bc -l <<< "$nmask/$nvar" | sed -e 's/^\./0\./g')

bashname="${VMV_name}_${1}"

gpu=0
input=$2

#CUDA_VISIBLE_DEVICES=<my_GPU_id> python3 DSAE_TORCH_ARG.py --input <my_input_file> --min_mask <my_min_mask> 

while read line; do

     echo -e $line | sed -e "s/<my_GPU_id>/$gpu/g" | sed -e "s~<my_input_file>~$input~g" | sed -e "s~<my_min_mask>~0.80~g" | sed -e "s~<my_max_mask>~$mrate~g"

     ((gpu=gpu+1))

     if [ $gpu -eq $n_gpus ]; then
        gpu=0
     fi

done < $1 > $bashname

echo -e "Training script generated at $bashname"


echo "example, parallel run automation:"
echo "split -l 1 -a 3 -d $bashname $bashname."
echo "for i in $bashname.sh.[0-9][0-9][0-9]; do echo "nohup bash $i 1> $i.out 2> $i.log"; done > run.sh"
echo "nohup parallel -j 4 < run.sh &"
echo "or run each line of run.sh as a parallel background process (add &) with nohup"

