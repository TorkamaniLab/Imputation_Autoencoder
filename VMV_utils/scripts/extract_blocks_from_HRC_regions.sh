#######################################################################################################
# SECOND PART, WILL BE EXECUTED BY THE USER                                                           #
# THIS WILL BE SEPARATED FROM FIRST PART ONCE WE HAVE GENERATED ALL chr${chr}_regions_file.txt FILES  #
# PLEASE SEPARATE THIS INTO ANOTHER SCRIPT AFTER HAVING ALL chr${chr}_regions_file.txt FILES          #
#######################################################################################################

rdir=/mnt/stsi/stsi5/raqueld/VMV_utils/regions

module load samtools

chr=$1

input=$2

if [ -z $chr ]; then
   echo "Usage example: bash extract_blocks_from_HRC_regions.sh 22 input_WGS.vcf.gz outdir"
   echo "Usage example2: bash extract_blocks_from_HRC_regions.sh 22 ARIC_chr22_ground_truth_5_phase/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-5.chr22.phased.vcf.gz AFR"
   exit
fi
if [ -z $input ]; then
   echo "Usage example: bash extract_blocks_from_HRC_regions.sh 22 input_WGS.vcf.gz outdir"
   echo "Usage example2: bash extract_blocks_from_HRC_regions.sh 22 ARIC_chr22_ground_truth_5_phase/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-5.chr22.phased.vcf.gz AFR"
   exit
fi

outdir=$3

if [ -z $outdir ]; then
    outdir="."
else
    if [ ! -d $outdir ]; then
        mkdir $outdir
    fi
fi

if [ -f $input ]; then
    if [ ! -f ${input}.tbi ]; then
        echo "tabix -p vcf $input"
        tabix -p vcf $input
    fi
else
    echo "File not found: $input"
    exit
fi

if [ ! -d $outdir/chr${chr} ]; then
    mkdir $outdir/chr${chr}
fi


echo $outdir/chr${chr}/run_${chr}.sh

while read region_line; do

    suffix=$(echo $region_line | cut -f 2 -d ' ')
    region=$(echo $region_line | cut -f 1 -d ' ')
    	
    filename=$(basename $input | sed -e "s/\.vcf\.gz/${suffix}/g" );

    echo $suffix $filename

    echo -e "bcftools view $input -r $region -Ov -o  $outdir/chr${chr}/$filename; bgzip -c $outdir/chr${chr}/$filename > $outdir/chr${chr}/$filename.gz; tabix -p vcf -f $outdir/chr${chr}/$filename.gz" >> $outdir/chr${chr}/run_${chr}.sh

done < $rdir/chr${chr}_regions_file.txt

parallel -j 16 < $outdir/chr${chr}/run_${chr}.sh
