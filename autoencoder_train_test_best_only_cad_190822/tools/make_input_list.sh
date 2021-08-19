input=/gpfs/home/sfchen/stsi1/201123_cosine_similarity/regions_all_split/cad_190822_regions.tsv
for i in $(tail -n +2 $input | cut -f 1,4 | tr '\t' '_'); do 
    chr=$(echo $i | tr '_' ' ' | awk '{print $1}'); 
    start=$(echo $i | tr '_' ' ' | awk '{print $2}'); 
    find /mnt/stsi/stsi5/raqueld/HRC/VMV/hg19/training/chr$chr | grep $start -m 2; 
done > cad_190822_file_list.txt

while read line; do ln -s $line /mnt/stsi/stsi5/raqueld/HRC/VMV/hg19/training/cad_190822_regions/; done < cad_190822_file_list.txt

