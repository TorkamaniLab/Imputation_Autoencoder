for i in chr*_masked/*.masked.gz; do o=$(echo $i | sed -e 's/\.gz$//g'); echo -e "zcat $i > $o"; done > extract_masked.sh
qsub -v cmd=extract_masked.sh preprocess.qsub -N extract_masked
