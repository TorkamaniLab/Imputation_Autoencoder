GROUND TRUTH (WGS):
chr{1..22}

PHASED AFFYMETRIX 6.0 GENOTYPE ARRAY (MASKED):
chr{1..22}_AFFY6

UNPHASED AFFYMETRIX 6.0 GENOTYPE ARRAY (MASKED):
chr{1..22}_AFFY6_unphased

PHASED IMPUTED (beagle, minimac, impute):
chr{1..22}_AFFY6_beagle5
chr{1..22}_AFFY6_impute5
chr{1..22}_AFFY6_minimac4

UNPHASED IMPUTED:
chr{1..22}_AFFY6_unphased_beagle5
chr{1..22}_AFFY6_unphased_impute5
chr{1..22}_AFFY6_unphased_minimac4



#HRC
#/mnt/stsi/stsi5/raqueld/HRC/ref_panel/hg19/HRC.r1-1.EGA.GRCh37.chr8.haplotypes.clean.vcf.gz

for i in {1..22}; do
    sbatch --export chr=${i},infile=/mnt/stsi/stsi5/raqueld/HRC/ref_panel/hg19/vcf/HRC.r1-1.EGA.GRCh37.chr${i}.haplotypes.clean.vcf.gz run_extractions.sbatch
done
for i in {1..22}; do echo -e "bash run_unphasing_WGS_SLURM.sh chr${i}"; done > parallel.sh; parallel -j 16 < parallel.sh

####ARIC#####################################
#DONT RUN ExtractionA or B
#run this command on the regions pre-extracted from HRC
while read line; do echo $line; $line; sleep 1; done < create_region_command_list.sh

#/mnt/stsi/stsi5/raqueld/ARIC/WGS/hg19/phased/

#always cp
mkdir AFR; cd AFR; cp /mnt/stsi/stsi5/raqueld/VMV_utils/scripts/* ./

for i in {1..22}; do
    sbatch --export chr=${i},infile=/mnt/stsi/stsi5/raqueld/ARIC/WGS/hg19/phased/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-5.chr${i}.phased.vcf.gz run_extractions.sbatch
done

cd ..; mkdir EUR; cd EUR; cp /mnt/stsi/stsi5/raqueld/VMV_utils/scripts/* ./

for i in {1..22}; do
    sbatch --export chr=${i},infile=/mnt/stsi/stsi5/raqueld/ARIC/WGS/hg19/phased/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1.chr${i}.phased.vcf.gz run_extractions.sbatch
done

cd ..; mkdir mixed; cd mixed; cp /mnt/stsi/stsi5/raqueld/VMV_utils/scripts/* ./

for i in {1..22}; do
    sbatch --export chr=${i},infile=/mnt/stsi/stsi5/raqueld/ARIC/WGS/hg19/phased/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-mixed.chr${i}.phased.vcf.gz run_extractions.sbatch
done

cd ..; mkdir EUR_398; cd EUR_398; cp /mnt/stsi/stsi5/raqueld/VMV_utils/scripts/* ./
for i in {1..22}; do bash downsample_EUR.sh $i; done

cd ..; mkdir EUR_AFR; cd EUR_AFR; cp /mnt/stsi/stsi5/raqueld/VMV_utils/scripts/* ./
for i in {1..22}; do bash merge.sh $i; done



###################Wellderly#########################
for i in {1..22}; do
    sbatch --export chr=${i},infile=/mnt/stsi/stsi5/raqueld/wellderly/WGS/hg19/phased/Wellderly.chrALL.g.lifted_hg19_to_GRCh37.GH.ancestry-1.chr${i}.phased.vcf.gz run_extractions.sbatch
done


############################HGDP#########################
#split into VMVs
#/mnt/stsi/stsi5/raqueld/HGDP/hg19_lifted_from_hg38
cp /mnt/stsi/stsi0/raqueld/ARIC_VMV/chr*_regions_file.txt ./
for i in {1..22}; do
    sbatch --export chr=${i},infile=/mnt/stsi/stsi5/raqueld/HGDP/WGS/hg19_from_hg38/picard_QC/hgdp_wgs.20190516.statphase.autosomes.chr${i}.hg19.concat.rename.bi.dense.vcf.gz run_extractions.sbatch
done

##############################AFAM####################
#split into VMVs
#/mnt/stsi/stsi5/raqueld/AFAM/hg19_lifted_from_hg38
cp /mnt/stsi/stsi0/raqueld/ARIC_VMV/chr*_regions_file.txt ./
for i in {1..22}; do
    sbatch --export chr=${i},infile=/mnt/stsi/stsi5/raqueld/AFAM/WGS/hg19_from_hg38/picard_QC/afam.n2303.chr${i}.refpanel.hg19.concat.rename.bi.dense.vcf.gz run_extractions.sbatch
done


#mask
for i in {1..22}; do sbatch --export chr=${i},rfile=AFFYMETRIX_6_positions.regions0,array=AFFY6 extract_genotype_array_from_WGS.sbatch; done
for i in {1..22}; do sbatch --export chr=${i},rfile=UKBB.regions,array=AXIOM extract_genotype_array_from_WGS.sbatch; done
for i in {1..22}; do sbatch --export chr=${i},rfile=OMNI2.5M.regions0,array=OMNI2M extract_genotype_array_from_WGS.sbatch; done
for i in {1..22}; do sbatch --export chr=${i},rfile=OMNI1M.regions0,array=OMNI1M extract_genotype_array_from_WGS.sbatch; done
for i in {1..22}; do sbatch --export chr=${i},rfile=OMNI5M.regions0,array=OMNI5M extract_genotype_array_from_WGS.sbatch; done

#unphase
sbatch unphasing_submitter.sbatch

#impute

sbatch minimac_submitter.sbatch
sbatch impute_submitter.sbatch
sbatch beagle_submitter.sbatch

#calculate accuracy
sbatch accuracy_calculator_submitter.sbatch
