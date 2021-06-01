
for chr in $1; do
    echo > run_jobs_SLURM_$chr.sh

    echo chr$chr
    mkdir chr$chr
    for i in ../EUR_398/chr$chr/*.VMV1; do
        name=$(basename $i | sed -e 's/ancestry-1\.//g')
        jname=$(basename $i  | sed -e 's/ancestry-1\./ancestry-5\./g')
        j="../AFR/chr$chr/$jname"
        cmd0="bgzip -c $i > $i.gz; tabix -f -p vcf $i.gz"
        cmd1="bgzip -c $j > $j.gz; tabix -f -p vcf $j.gz"
        cmd2="bcftools merge $i.gz $j.gz -m none -Ov -o chr$chr/$name"
        cmd3="bgzip -c chr$chr/$name > chr$chr/$name.gz"
        cmd4="tabix -f -p vcf chr$chr/$name.gz"
        #echo "$cmd0; $cmd1; $cmd2; $cmd3; $cmd4"
        echo "$cmd2; $cmd3; $cmd4"
    done > run_merge_chr$chr.sh
    echo "sbatch --export cmd=run_merge_chr$chr.sh preprocess.sbatch" >> run_jobs_SLURM_$chr.sh
    bash run_jobs_SLURM_$chr.sh
done

