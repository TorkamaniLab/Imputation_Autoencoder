#/mnt/stsi/stsi0/raqueld/keras_tuner/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.50350213-50935103.m414.1_50350235-50491042.VMV1


#module load python/3.6.3
module load python/3.8.3
module load samtools

#indir=/mnt/stsi/stsi0/raqueld/keras_tuner/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_masked_MAX
#outdir=/mnt/stsi/stsi0/raqueld/keras_tuner/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_masked_MAX_unphased

#indir=/mnt/stsi/stsi0/raqueld/keras_tuner/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_AFFY6
#outdir=/mnt/stsi/stsi0/raqueld/keras_tuner/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_AFFY6_unphased

indir=$(readlink -e $1)
outdir=${indir}_unphased
script=/gpfs/home/raqueld/DSAE_TF2_grid_search/unphase/unphase_GT_VMV-ARG.py


if [ -z $1 ]; then
    echo "usage: bash run_unphasing.sh <indir>"
    echo "example: bash run_unphasing.sh /mnt/stsi/stsi0/raqueld/keras_tuner/ARIC/ARIC_chr22_ground_truth_5_phase_VMV_376a1_376a5_merged_AFFY6"
    exit
fi

echo "Output directory at $outdir"

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

cp run_unphase.sbatch $outdir/
cd $outdir

for i in $indir/*.VMV1.gz; do

    name=$(basename $i | sed -e 's/\.gz$//g')
    cmd="python3 $script $i $outdir/$name; bgzip -c $outdir/$name > $outdir/$name.gz; tabix -f -p vcf $outdir/$name.gz"
    echo $cmd
    #$cmd
done > run_unphase.sh

echo "sbatch --export cmd=run_unphase.sh run_unphase.sbatch"
sbatch --export cmd=run_unphase.sh run_unphase.sbatch

exit

