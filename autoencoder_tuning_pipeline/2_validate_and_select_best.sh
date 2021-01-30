#sbatch --export=mdir=chr22/22_17274081-17382360,input=input.cfg --job-name=2_validate_and_select_best 2_validate_and_select_best.sbatch

mdir=$(readlink -e $1)
input=$(readlink -e $2)

gsstarttime=$(date +%s)

echo -e "bash 2_generate_validation_commands.sh $mdir $input"

bash 2_generate_validation_commands.sh $mdir $input > $mdir/validation_commands.sh

echo -e "cd $mdir\nbash validation_commands.sh"
cd $mdir
bash validation_commands.sh

gsendtime=$(date +%s)

gsruntime=$((gsendtime-gsstarttime))


echo "Validation run time: $gsruntime"

