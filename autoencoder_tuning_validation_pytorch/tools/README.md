Estimates the amount of VRAM or GPU memory usage required for training a model.
Usage:
```
python3 estimate_VRAM_needed_for_autoencoder.py

Usage: CUDA_VISIBLE_DEVICES=<GPUID> python3 estimate_VRAM_needed_for_autoencoder.py <N_VARIANTS>
    <GPUID>: GPU ID or GPU index (i.e. 0)
    <N_VARIANTS>: number of genetic variants in the VMV/VCF file
```

More realistic example, for a VCF with 3600 variants:
```
python3 estimate_VRAM_needed_for_autoencoder.py 3600

SIMULATION PARAMETERS (WORST CASE SCENARIO FROM GRID SEARCH)
number of variants: 3600
number of input/output nodes: 7200
batch size: 256
number of layers: 8
size ratio: 1.0
Optimizer: Adam
GPU RAM for pytorch session only: 9032.0MiB
GPU RAM including extra driver buffer from nvidia-smi: 10285MiB
```
