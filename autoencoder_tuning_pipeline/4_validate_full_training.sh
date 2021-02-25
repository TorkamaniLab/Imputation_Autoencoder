#sbatch --export=mdir=chr22/22_17274081-17382360,input=input.cfg --job-name=4_validate_full_training 4_validate_full_training.sbatch

mdir=$1
input=$2

gsstarttime=$(date +%s)

echo -e "bash 2_generate_validation_commands.sh $mdir $input best_only  > $mdir/validation_commands.sh"

bash 2_generate_validation_commands.sh $mdir $input best_only > $mdir/validation_commands.sh

echo -e "cd $mdir\nbash validation_commands.sh"
cd $mdir
bash validation_commands.sh

echo -e "If job is successful, results will be at: $mdir/plots_full_training_best"

best=$(cat $mdir/full_training_plots_*/overall_results_per_model.tsv |  grep -v Mean_r2 | grep -v ^phased | sort -k2,2g | tail -n 1 | cut -f 1)
found=$(echo $best | grep "^model" | wc -l)

if [ $found -eq 0 ]; then
    bi=$(cat BATCH_ID | tr '.' ' ' | awk '{print $NF}')
    new_bi=$(($bi+1))
    new_suffix=$(printf %03d $a)
    sed -i -e "s/$bi$/$new_suffix/g" BATCH_ID
    echo -e "Best model is $best, keeping VMV in the list for next grid search iteration."
    bID=$(cat BATCH_ID)
    echo -e "New hyperparameter set is $mdir/$bID"
else
    bID=$(cat BATCH_ID)
    echo -e "Best model is $best, no need to run additional search. Grid search converged at hyperparameter set $bID"
    mroot=$(dirname $mdir)
    region=$(basename $mdir)
    for i in $mroot/full_training_list.txt*; do
        sed -i -e "/$region$/d" $i
    done
    sed -i -e "/$region$/d" $mroot/model_dir_list.txt
    echo -e "Removed $region from $mroot/full_training_list.txt"
    echo -e "Removed $region from $mroot/model_dir_list.txt"
fi

gsendtime=$(date +%s)

gsruntime=$((gsendtime-gsstarttime))


echo "Validation run time: $gsruntime"
