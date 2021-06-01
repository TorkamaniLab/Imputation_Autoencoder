

script="~/Compare_imputation_to_WGS.py"

chr=$1

array=$2

#python3 imputation_accuracy_calculator/Compare_imputation_to_WGS.py
#--ga /mnt/stsi/stsi0/raqueld/keras_tuner/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_masked/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.17274081-17382360.vcf.VMV1.masked.gz
#--wgs /mnt/stsi/stsi0/raqueld/keras_tuner/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.17769623-17825562.vcf.VMV1.gz
#--imputed c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.17274081-17382360.vcf.VMV1.masked.imputed_minimac.dose.vcf.gz
#--ref /mnt/stsi/stsi0/raqueld/HRC/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.clean.vcf.gz

#ga_dir="/mnt/stsi/stsi0/raqueld/keras_tuner/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_AFFY6"
#wgs_dir="/mnt/stsi/stsi0/raqueld/keras_tuner/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged"
#imputed_dir="/mnt/stsi/stsi0/raqueld/keras_tuner/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_AFFY6_minimac"

wgs_dir=chr$chr
ga_dir=chr${chr}_$array
#imputed_dirs="chr${chr}_masked_minimac4 chr${chr}_masked_beagle5 chr${chr}_masked_impute5 chr${chr}_masked_unphased_minimac4 chr${chr}_masked_unphased_beagle5 chr${chr}_masked_unphased_impute5"

imputed_dirs="chr${chr}_${array}_minimac4 chr${chr}_${array}_beagle5 chr${chr}_${array}_impute5 chr${chr}_${array}_unphased_minimac4 chr${chr}_${array}_unphased_beagle5 chr${chr}_${array}_unphased_impute5"

#imputed_dirs="chr${chr}_AXIOM_minimac4 chr${chr}_AXIOM_beagle5 chr${chr}_AXIOM_impute5 chr${chr}_AXIOM_unphased_minimac4 chr${chr}_AXIOM_unphased_beagle5 chr${chr}_AXIOM_unphased_impute5"
#imputed_dirs="chr${chr}_AFFY6_minimac4 chr${chr}_AFFY6_beagle5 chr${chr}_AFFY6_impute5 chr${chr}_AFFY6_unphased_minimac4 chr${chr}_AFFY6_unphased_beagle5 chr${chr}_AFFY6_unphased_impute5"

echo -e "dir\tgafile\tnvar" > missing_imputed_files_for_chr$chr.log

for i in $wgs_dir/*.VMV1.gz; do

    suffix=$(basename $i | sed -e 's/.*\.phased\.//g' | sed -e 's/\.vcf\.VMV1.*/\.vcf\.VMV1/g' | sed -e 's/\.VMV1.*/\.VMV1/g')

    ga=$(find $ga_dir -name *$suffix.masked.gz)
    wgs=$(find $wgs_dir -name *$suffix.gz)
    for imputed_dir in $imputed_dirs; do

        imputed=$(find $imputed_dir -name *$suffix.*.vcf.gz)
        if [ ! -z $imputed ]; then
            cmd="python3 $script --wgs $wgs --imputed $imputed --ga $ga"

            #echo $ga
            #echo $wgs
            #echo $imputed
            echo $cmd
        else
            nvars=$(zcat $ga | grep -v "#" | wc -l)
            echo -e "$imputed_dir\t$ga\t$nvars" >> missing_imputed_files_for_chr$chr.log

        fi
    done


done > chr$chr/run_calculator_${chr}_${array}.sh

#exit

split -l 2000 -a 3 -d chr$chr/run_calculator_${chr}_${array}.sh chr$chr/run_calculator_${chr}_${array}.sh.

for i in chr$chr/run_calculator_${chr}_${array}.sh.*; do

    echo "sbatch --export=cmd=$i --job-name=$i run_calculator.sbatch"

done > chr$chr/submit_calculator_jobs_${chr}_${array}.sh

#exit

while read line; do

    echo $line
    $line
    sleep 2

done < chr$chr/submit_calculator_jobs_${chr}_${array}.sh
