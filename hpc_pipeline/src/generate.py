#!/usr/bin/python3
#
# Notes:
# * original kept list of model output in directories
#    in $out_root/model_dir_list.txt
# * and the experiment list (${out_root}/*/*_train.sh.*) was stored in
#    ${out_root}/full_training_list.txt

import sys
import gzip
import glob
import yaml
from pathlib import Path
from typing import Optional, List, Dict

#from configs import Tile, Config, read_config
from genomeai import Tile, Config, read_config
from pydantic import BaseModel

from name import parse_name


def glob_search(base, name):
    ans = []
    for fname in glob.glob(f"{base}/**.gz"):
        if name in fname:
            ans.append(fname)
    if len(ans) != 1:
        print(base)
        raise ValueError(f"Found zero or multiple validation sets: {ans}")
    return ans[0]

def launch_cmd(c : Config, e : Tile):
    """ Create the hyperparameter tuning launch command for this tile.

    Notes:

    * The original intent was to paste 6 of these back-to-back inside a job file
      so that 600 training trials are run.
    * This is transitional, since ideally the experiment will read its
      config. file directly.
    """

    ga_list = " ".join(f"--val_input {v}" for v in e.validation)
    wgs_list = " ".join(f"--val_true {g}" for g in e.ground)
    sql = "--mysql {c.mysql}" if c.mysql else ""
    return f"""
CUDA_LAUNCH_BLOCKING=1 python3 DSAE_TORCH_ARG_PHASED.py --min_mask 0.80 --max_mask {e.mrate} \\
        --study_name {e.name} --n_trials {c.trials_per_job} --sampler {c.sampler} --patience {c.patience} \\
        --sampling_res {c.sampling_res} --pruning {int(c.pruning)} --max_models_per_gpu {c.max_models_per_gpu} \\
        --resume 1 --input {e.input_file} {ga_list} {wgs_list} {sql}
"""

def count_records(fname):
    """Return the number of non-comment lines in the file.
    """
    rec = 0
    with gzip.open(fname, 'rt') as f:
        for line in f:
            if line.strip()[0] != '#':
                rec += 1
    return rec

def main(argv):
    assert len(argv) == 3, f"Usage: {argv[0]} <config.yaml> <out_dir>"

    # read configuration options
    config = read_config(argv[1])
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    
    i=1
    j=1
    print(f"tile{i}")
    targets = open("targets.yaml", "a")

    # loop through experiments and write one Tile each
    for vmv in glob.glob( str(Path(config.train_dir) / '*.VMV1.gz') ):
        # parse chr, region from filename
        a = parse_name(vmv)
        source, revision, ega, dataset, chromosome, start, end = \
                a['source'], a['revision'], a['ega'], a['dataset'], a['chromosome'], a['start'], a['end']
        #VMV_name = Path(vmv).stem.stem
        #print(VMV_name)
        nvar = count_records(vmv)
        mrate = (nvar-5.0) / nvar

        region = f'{start}-{end}'
        E = Tile( name=f"{chromosome}_{start}-{end}"
                      , input_file=vmv
                      , dataset=dataset
                      , source=source
                      , revision=revision
                      , chromosome=chromosome
                      , start=start
                      , end= (end if end != '' else None)
                      , nvar=nvar
                      , mrate=mrate
                      , validation=[glob_search(v.format(chromosome=chromosome), region) for v in config.val_ga_dir]
                     , ground=[glob_search(v.format(chromosome=chromosome), region) for v in config.val_wgs_dir]
                     )
        assert len(E.validation) > 0, f"Can't find validation data for {vmv}"
        assert len(E.ground) > 0, f"Can't find ground data for {vmv}"

        #print(E)
        #print(launch_cmd(config, E))

        mdir = out / f"{chromosome}_{start}-{end}" # {chr}_{region}
        mdir.mkdir(parents=True, exist_ok=True)
        with open(mdir / "tile.yaml", 'w') as f:
            yaml.dump(E.dict(), f)
        
        dirname = E.name
        if E.nvar <= 2000:

        targets.write(
            f"tile{i+j-1}:\n  dirname: {dirname}\n  more-flags:\n\n")
        i += 1

    targets.close()
    targets_hm.close()

if __name__=="__main__":
    main(sys.argv)
