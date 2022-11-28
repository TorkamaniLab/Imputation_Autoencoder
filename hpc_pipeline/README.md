# Genome Imputation on HPC Processing Scripts

Steps to run genome imputation for a chromosome using this HPC
pipeline.  An installed version of the `genomeai` package is required.

This process starts from the following data directories:

Training data: '/gpfs/alpine/proj-shared/bif138/HRC/VMV/hg19/training/chr22'

Validation data:
  - '/gpfs/alpine/proj-shared/bif138/ARIC/VMV/hg19/validation/EUR_AFR/chr{chromosome}/chr{chromosome}_AFFY6'
  - '/gpfs/alpine/proj-shared/bif138/ARIC/VMV/hg19/validation/EUR_AFR/chr{chromosome}/chr{chromosome}_AXIOM'
  - '/gpfs/alpine/proj-shared/bif138/ARIC/VMV/hg19/validation/EUR_AFR/chr{chromosome}/chr{chromosome}_OMNI1M'

Ground truth: '/gpfs/alpine/proj-shared/bif138/ARIC/VMV/hg19/validation/EUR_AFR/chr{chromosome}/chr{chromosome}'

All three of these data sources contain a series of "tiles".  Each "tile"
is a vmv formatted file containing SNP data for a contiguous region of
a single chromosome.  The region sizes vary, but the count of variants
within the tile is usually around 500-3000 SNPs.

Then the `src/generate.py` program is run.
It creates a single directory for each tile it finds in the training dataset.
Every directory contains a `tile.yaml` file containing metadata
summarizing the properties of the tile.
The other output of `generate.py` are the `targets.yaml` and `targets_hm.yaml`
files.  Both list tile names to be processed.  Each tile name in that file
further contains a list of required output information we would like
to build within each directory.

Next, the `pmake` program is run repeatedly within batch allocations to
build those outputs in parallel.

