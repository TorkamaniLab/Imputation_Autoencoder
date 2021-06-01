#bash extract_genotype_array_from_WGS.sh 22 AFFYMETRIX_6_positions.regions0 AFFY6
#rfile=AFFYMETRIX_6_positions.regions0

#array=AFFY6

rfile=$2

array=$3

outdir=chr${1}_${array}

module load samtools
if [ ! -d $outdir ]; then
    mkdir $outdir

fi

oscript=$outdir/extract.sh
echo > $oscript

for i in chr${1}/*.VMV1; do 
    echo $i; j=$(basename $i); 

    echo -e "bcftools view $i.gz -R $rfile -Oz -o $outdir/$j.masked.gz; tabix -f -p vcf $outdir/$j.masked.gz;" >> $oscript
done

parallel -j 16 < $oscript
