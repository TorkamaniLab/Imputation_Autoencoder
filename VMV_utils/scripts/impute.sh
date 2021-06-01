bin=/gpfs/home/sfchen/bin/impute_v5/impute5

chr=$1

map=/mnt/stsi/stsi5/raqueld/maps/shapeit4_map/chr$chr.b37.gmap.gz
refdir=/mnt/stsi/stsi5/raqueld/HRC/VMV/hg19/imputation_ref/phased_impute5/chr${chr}

    indir="chr${chr}_$2"
    outdir="${indir}_impute5"
    [ ! -d $outdir ] && mkdir $outdir

    for i in $indir/*.masked.gz; do

        region=$(basename $i | sed -e 's/.*\.phased\.//g' | sed -e 's/.*\.dense\.//g' | sed -e 's/\.VMV1\..*//g')
        #echo $region 
        ref=$(find $refdir -name *.${region}.VMV1.vcf.gz)
        out=$(basename $i | sed -e 's/\.gz/\.imputed_impute.vcf.gz/g')
        cmd0="$bin --h $ref --g $i --o $outdir/$out --r $chr --m $map --b 1000 && tabix -f -p vcf $outdir/$out; tabix -p vcf -f $outdir/$out"
        echo "$cmd0;"
    done > $outdir/run_impute_chr${chr}_$2.sh

    echo "sbatch --export cmd=$outdir/run_impute_chr${chr}_$2.sh preprocess.sbatch" > $outdir/run_imputation_jobs_SLURM_${1}_$2.sh
    bash $outdir/run_imputation_jobs_SLURM_${1}_$2.sh

refdir=/mnt/stsi/stsi5/raqueld/HRC/VMV/hg19/imputation_ref/unphased_impute5/chr${chr}

    indir="chr${chr}_$2_unphased"
    outdir="${indir}_impute5"
    [ ! -d $outdir ] && mkdir $outdir
        

    for i in $indir/*.masked.gz; do

        region=$(basename $i | sed -e 's/.*\.phased\.//g' | sed -e 's/.*\.dense\.//g' | sed -e 's/\.VMV1\..*//g')
        #echo $region 
        ref=$(find $refdir -name *.${region}.VMV1.vcf.gz)
        out=$(basename $i | sed -e 's/\.gz/\.imputed_impute.vcf.gz/g')
        cmd1="$bin --h $ref --g $i --o $outdir/$out --r $chr --m $map --b 1000 && tabix -f -p vcf $outdir/$out; tabix -p vcf -f $outdir/$out"
        echo "$cmd1;"
    done > $outdir/run_impute_unphased_chr${chr}_$2.sh
    echo "sbatch --export cmd=$outdir/run_impute_unphased_chr${chr}_$2.sh preprocess.sbatch" > $outdir/run_imputation_jobs_SLURM_${1}_$2.sh
    bash $outdir/run_imputation_jobs_SLURM_${1}_$2.sh    
