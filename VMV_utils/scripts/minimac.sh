#!/bin/bash

#bash minimac.sh 22 AXIOM

chr=$1
#/mnt/stsi/stsi5/raqueld/HRC/VMV/hg19/imputation_ref
refdir=/mnt/stsi/stsi5/raqueld/HRC/VMV/hg19/imputation_ref/phased_minimac4/chr${chr}

indir=chr${chr}_$2
outdir="./${indir}_minimac4"
[ ! -d $outdir ] && mkdir $outdir

    for i in $indir/*.masked.gz; do

        region=$(basename $i | sed -e 's/.*\.phased\.//g' | sed -e 's/.*\.dense\.//g' | sed -e 's/\.VMV1\..*//g')
        #echo $region 
        ref=$(find $refdir -name *.${region}.*VMV1_ref.m3vcf.gz)
        out=$(basename $i | sed -e 's/\.gz/\.imputed_minimac/g')
        cmd0="~/bin/minimac4/minimac4 --refHaps $ref --haps $i --prefix $outdir/$out --cpus 16 --minRatio 0.01; tabix -p vcf -f $outdir/$out.dose.vcf.gz"
        echo "$cmd0;"
    done > $outdir/run_minimac_chr${chr}_$2.sh
    echo "sbatch --export cmd=$outdir/run_minimac_chr${chr}_$2.sh preprocess.sbatch" > $outdir/run_imputation_jobs_SLURM_${chr}_$2.sh

bash $outdir/run_imputation_jobs_SLURM_${chr}_$2.sh

refdir=/mnt/stsi/stsi5/raqueld/HRC/VMV/hg19/imputation_ref/unphased_minimac4/chr${chr}

indir=chr${chr}_$2_unphased
outdir="./${indir}_minimac4"
[ ! -d $outdir ] && mkdir $outdir

    for i in $indir/*.masked.gz; do

        region=$(basename $i | sed -e 's/.*\.phased\.//g' | sed -e 's/.*\.dense\.//g' | sed -e 's/\.VMV1\..*//g')
        #echo $region 
        ref=$(find $refdir -name *.${region}.*VMV1_ref.m3vcf.gz)
        out=$(basename $i | sed -e 's/\.gz/\.imputed_minimac/g')
        cmd1="~/bin/minimac4/minimac4 --refHaps $ref --haps $i --prefix $outdir/$out --cpus 16 --minRatio 0.01; tabix -p vcf -f $outdir/$out.dose.vcf.gz"
        echo "$cmd1;"
    done > $outdir/run_minimac_unphased_chr${chr}_$2.sh
    echo "sbatch --export cmd=$outdir/run_minimac_unphased_chr${chr}_$2.sh preprocess.sbatch" > $outdir/run_imputation_jobs_SLURM_${chr}_$2.sh
    
bash $outdir/run_imputation_jobs_SLURM_${chr}_$2.sh
