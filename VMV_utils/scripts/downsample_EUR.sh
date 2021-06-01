#example, after you extract EUR
#mkdir EUR_398
#cd EUR_398
#bash downsample_EUR.sh

n=398
((n=n+9))

for chr in $1; do
    echo > run_jobs_SLURM_$chr.sh

    indir="../EUR/chr$chr"
    outdir=$(basename $indir)
    mkdir $outdir
    echo $indir

    for i in $indir/*.VMV1; do

        name=$(basename $i)

        echo "cat $i | cut -f1-$n > $outdir/$name; bgzip -c $outdir/$name > $outdir/$name.gz; tabix -p vcf -f $outdir/$name.gz"

    done > ${outdir}_subsample.sh
    echo "sbatch --export cmd=${outdir}_subsample.sh preprocess.sbatch" >> run_jobs_SLURM_$chr.sh
    bash run_jobs_SLURM_$chr.sh 
done

