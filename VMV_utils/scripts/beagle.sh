#bin=/gpfs/home/sfchen/bin/beagle/beagle.12Jul19.0df.jar
bin=/gpfs/home/raqueld/bin/beagle.21Apr21.304.jar

chr=$1

map=/mnt/stsi/stsi5/raqueld/maps/beagle_map/plink.chr$chr.GRCh37.map

refdir=/mnt/stsi/stsi5/raqueld/HRC/VMV/hg19/imputation_ref/phased_beagle5/chr${chr}

   indir="chr${chr}_$2"
   outdir="${indir}_beagle5"
   [ ! -d $outdir ] && mkdir $outdir

    for i in $indir/*.masked.gz; do

        region=$(basename $i | sed -e 's/.*\.phased\.//g' | sed -e 's/.*\.dense\.//g' | sed -e 's/\.VMV1\..*//g')
        #echo $region 
        ref=$(find $refdir -name *.${region}.*VMV1_ref.bref3)
        out=$(basename $i | sed -e 's/\.gz/\.imputed_beagle/g')
        cmd0="java -jar $bin ref=$ref gt=$i out=$outdir/$out chrom=$chr map=$map; tabix -p vcf -f $outdir/$out.vcf.gz"
        echo "$cmd0;"
    done > $outdir/run_beagle_chr${chr}_$2.sh
    echo "sbatch --export cmd=$outdir/run_beagle_chr${chr}_$2.sh preprocess.sbatch" > $outdir/run_imputation_jobs_SLURM_${1}_${2}.sh
    bash $outdir/run_imputation_jobs_SLURM_${1}_${2}.sh

refdir=/mnt/stsi/stsi5/raqueld/HRC/VMV/hg19/imputation_ref/unphased_beagle5/chr${chr}

   indir="chr${chr}_$2_unphased"
   outdir="${indir}_beagle5"
   [ ! -d $outdir ] && mkdir $outdir

    for i in $indir/*.masked.gz; do

        region=$(basename $i | sed -e 's/.*\.phased\.//g' | sed -e 's/.*\.dense\.//g' |  sed -e 's/\.VMV1\..*//g')
        #echo $region 
        ref=$(find $refdir -name *.${region}.*VMV1_ref.bref3)
        out=$(basename $i | sed -e 's/\.gz/\.imputed_beagle/g')
        cmd1="java -jar $bin ref=$ref gt=$i out=$outdir/$out chrom=$chr map=$map; tabix -p vcf -f $outdir/$out.vcf.gz"
        echo "$cmd1;"
    done > $outdir/run_beagle_unphased_chr${chr}_$2.sh
    echo "sbatch --export cmd=$outdir/run_beagle_unphased_chr${chr}_$2.sh preprocess.sbatch" > $outdir/run_imputation_jobs_SLURM_${1}_${2}.sh
    bash $outdir/run_imputation_jobs_SLURM_${1}_${2}.sh
