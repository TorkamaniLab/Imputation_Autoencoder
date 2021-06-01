#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --partition=shared
#SBATCH --time=600:00:00
#SBATCH --job-name=rsync_VMVs
#SBATCH --output=%x.oe%j
#SBATCH --error=%x.oe%j


rsync -rtv /mnt/stsi/stsi0/raqueld/ARIC_VMV/EUR_AFR/* /mnt/stsi/stsi5/raqueld/VMVs/ARIC/EUR_AFR/
rsync -rtv /mnt/stsi/stsi0/raqueld/wellderly_VMV/EUR_new/* /mnt/stsi/stsi5/raqueld/VMVs/wellderly/EUR/
rsync -rtv /mnt/stsi/stsi0/raqueld/HGDP_VMV/ALL/* /mnt/stsi/stsi5/raqueld/VMVs/HGDP/ALL/
rsync -rtv /mnt/stsi/stsi0/raqueld/afam_VMV/ALL/* /mnt/stsi/stsi5/raqueld/VMVs/afam_VMV/ALL/
rsync -rtv /mnt/stsi/stsi0/raqueld/VMV_VCF_Extractions/* /mnt/stsi/stsi5/raqueld/VMVs/HRC/ALL/
