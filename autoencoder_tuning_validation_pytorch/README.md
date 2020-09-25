## Dependencies

- Python >=3.6
- Pytorch >=1.6 (tested on 1.08 as well)
- Samtools >=1.9

## Example files:

The example files used in this tutorial have restricted access so they were not uploaded to github.
The files can be copied from garibaldi's login01 directory at: 
```
/mnt/stsi/p1/Torkamani_Lab/Internal/raqueld/example_files/autoencoder_pytorch/examples.tar.gz
```

Copy the example files into the cloned repository folder by:
```
cd <this repository dir>
cp /mnt/stsi/p1/Torkamani_Lab/Internal/raqueld/example_files/autoencoder_pytorch/examples.tar.gz ./
tar -xvf examples.tar.gz
```

## 1. Making random hyperparameter combination sets

To generate a subset of N hyperparameters:
```
python3 pytorch_random_grid_maker.py <N>
```

For example (100 hyperparameters):
```
python3 pytorch_random_grid_maker.py 100
```

The results are 100_random_hyperparameters.tsv (hyperparameter table) and 100_random_hyperparameters.sh (command templates).
Use the generated output command template file (100_random_hyperparameters.sh) to build your command, just replacing <my_GPU_id>,  <my_input_file>,  <my_min_mask>, <my_max_mask> by their respective desired values. 
For example, this line from 100_random_hyperparameters.sh:
```
CUDA_VISIBLE_DEVICES=<my_GPU_id> python3 DSAE_TORCH_ARG.py --input <my_input_file> \
    --min_mask <my_min_mask> --max_mask <my_max_mask> --model_id model_1 \
    --l1 1e-07 --l1 1e-08 --beta 0.0001 --rho 0.05 --gamma 0.0 --disable_alpha 1 \
    --learn_rate 0.001 --activation leakyrelu --optimizer rmsprop --loss_type FL \
    --n_layers 8 --size_ratio 0.7 --decay_rate 0.5
```

Would be replaced like this, for aiming on GPU 0, my_VMV_file.vcf as input, minimum mask of 0.8, and maximum mask of 0.99:
```
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input my_VMV_file.vcf \
    --min_mask 0.8 --max_mask 0.99 --model_id model_1 \
    --l1 1e-07 --l1 1e-08 --beta 0.0001 --rho 0.05 --gamma 0.0 \
    --disable_alpha 1 --learn_rate 0.001 --activation leakyrelu \
    --optimizer rmsprop --loss_type FL --n_layers 8 --size_ratio 0.7 --decay_rate 0.5
```

I made a simple bash helper script that will do this replacement automatically:
```
usage: bash make_training_script_from_template.sh <template.sh> <input.vcf> <max_gpus>
example: bash make_training_script_from_template.sh 100_random_hyperparameters.sh examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 4
bash make_training_script_from_template.sh 100_random_hyperparameters.sh examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 4
Training script generated at HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1_100_random_hyperparameters.sh
```


## 2. Train the models

To run training, just execute the script DSAE_TORCH_ARG.py following the example listed bellow. For quick testing, only --input, --min_mask, and --max_mask are required (the hyperparameters will then be set to their default values).
For example:
```
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input my_VMV_file.vcf --min_mask 0.8 --max_mask 0.99
```
Or a more realistic example:
```
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH.py \
--input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 \
--min_mask 0.80 --max_mask 0.99755620723362658846
```

After debugging, making sure it runs, you can play with the hyperparamters:
```
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH.py \
--input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 \
--min_mask 0.80 --max_mask 0.99755620723362658846 --model_id best_model \
--l1 1e-07 --l1 1e-08 --beta 0.0001 --rho 0.05 --gamma 0.0 --disable_alpha 1 \
--learn_rate 0.001 --activation leakyrelu --optimizer rmsprop --loss_type FL \
--n_layers 8 --size_ratio 0.7 --decay_rate 0.5
```

The outputs will be trained model and parameters (used for inference later), listed in the output folder path (which can be customized as well):
```
ls examples/IMPUTATOR_HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1/
best_model_param.py  best_model.pth
```

Repeat these steps for all the different hyperparameters you want to test and all VCFs you want to train, making sure you change --model_id argment value to not overwrite your previous model.

For more details use --help:
```
python3 DSAE_TORCH_ARG.py --help
Using tabix at: /usr/bin/tabix
usage: DSAE_TORCH_ARG.py [-h] [-I INPUT] [-N MIN_MASK] [-M MAX_MASK] [-L L1]
                         [-W L2] [-B BETA] [-R RHO] [-G GAMMA]
                         [-A DISABLE_ALPHA] [-C LEARN_RATE] [-F ACTIVATION]
                         [-O OPTIMIZER] [-T LOSS_TYPE] [-D N_LAYERS]
                         [-S SIZE_RATIO] [-E DECAY_RATE] [-H MODEL_ID]
                         [-J MODEL_DIR]


optional arguments:
  -h, --help            show this help message and exit
  -I INPUT, --input INPUT
                        [str] Input file (ground truth) in VCF format
  -N MIN_MASK, --min_mask MIN_MASK
                        [float] Minimum masking ratio
  -M MAX_MASK, --max_mask MAX_MASK
                        [float] Maximum masking ratio
  -L L1, --l1 L1        [float] L1 regularization scaling factor
  -W L2, --l2 L2        [float] L2 regularization scaling factor (a.k.a.
                        weight decay)
  -B BETA, --beta BETA  [float] Beta scaling factor for sparsity loss (KL
                        divergence)
  -R RHO, --rho RHO     [float] Rho desired mean activation for sparsity loss
                        (KL divergence)
  -G GAMMA, --gamma GAMMA
                        [float] gamma modulating factor for focal loss
  -A DISABLE_ALPHA, --disable_alpha DISABLE_ALPHA
                        [0 or 1]=[false or true] whether disable alpha scaling
                        factor for focal loss
  -C LEARN_RATE, --learn_rate LEARN_RATE
                        [float] learning rate
  -F ACTIVATION, --activation ACTIVATION
                        [relu, leakyrelu, tanh, sigmoid] activation function
                        type
  -O OPTIMIZER, --optimizer OPTIMIZER
                        [adam, sgd, adadelta, adagrad] optimizer type
  -T LOSS_TYPE, --loss_type LOSS_TYPE
                        [CE or FL] whether use CE for binary cross entropy or
                        FL for focal loss
  -D N_LAYERS, --n_layers N_LAYERS
                        [int, even number] total number of layers
  -S SIZE_RATIO, --size_ratio SIZE_RATIO
                        [float(0-1]] size ratio for successive layer shrink
                        (current layer size = previous layer size *
                        size_ratio)
  -E DECAY_RATE, --decay_rate DECAY_RATE
                        [float[0-1]] learning rate decay ratio (0 = decay
                        deabled)
  -H MODEL_ID, --model_id MODEL_ID
                        [int/str] model id or name to use for saving the model
  -J MODEL_DIR, --model_dir MODEL_DIR
                        [str] path/directory to save the model
```

## 3. Model validation and evaluation

### 3.1. Running inference

For evaluating the model we must run inference on an independent dataset first, for example:
```
python3 inference_function.py pos_file.1-5  genotype_array_file.vcf model_dir --output output_name.vcf --model_name model_name --use_gpu
```

A more specific example, on ARIC:
```
cat examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 | grep -v "#" | cut -f1-5 > examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1.1-5
python3 inference_function.py examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1.1-5 \
    examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked \
    examples/IMPUTATOR_HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 \
    --output examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked.gz.best_model.vcf \
    --model_name best_model --use_gpu
```

The result is the imputed VCF with the name you specified in the --output argument (e.g. examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked.gz.best_model.vcf).

For additional info run the help mode with --help flag:
```
python3 inference_function.py --help
Using tabix at: /usr/bin/tabix
usage: inference_function.py [-h] --model_name model_name [--output output]
                             [--use_gpu] [--debug]
                             reference.1-5 genotype_array model_dir


Imputation Inference Script.


positional arguments:
  reference.1-5         First 5 columns of reference panel VCF file
                        (chromosome position rsID REF ALT), used to build
                        imputed file
  genotype_array        Genotype array file in VCF format, file to be imputed
  model_dir             Pre-trained model directory path (just directory path,
                        no file name and no extension required


optional arguments:
  -h, --help            show this help message and exit
  --model_name model_name
                        model name to load which is located under model_dir ,
                        for multi-model support
  --output output       (Optional) An output file location (can be either a
                        file or a directory), imputed file in VCF format, same
                        name prefix as input if no out name is provided
  --use_gpu             Whether or not to use the GPU (default=False)
  --debug               Add predicted accuracies to the output for debugging
                        (default=False)
```

### 3.2. Run evaluation of the imputed results

Compress model and tabix it first, for example:
```
bgzip -c examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked.gz.best_model.vcf > examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked.gz.best_model.vcf.gz
tabix -f -p examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked.gz.best_model.vcf.gz
```
Then run the accuracy calculator, for example:
```
python3 Compare_imputation_to_WGS.py --wgs pos_file.1-5 --imputed imputed_file.vcf \
    --ref ref_file.vcf --ga ga_file.vcf --sout per_sample_output.tsv --vout per_variant_output.tsv
```
More specific example:
```
python3 Compare_imputation_to_WGS.py \
    --wgs examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.gz  \
    --imputed  examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked.gz.best_model.vcf.gz \
    --ref examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1.gz \
    --ga examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked.gz \
    --sout examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked.gz.best_model.vcf.gz_per_sample_tsv \
    --vout examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked.gz.best_model.vcf.gz_per_variant.tsv
```

Results will show...

Results per sample at: 
```
examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked.gz.best_model.vcf.gz_per_sample_tsv
```

Results per variant at: 
```
examples/c1_ARIC_WGS_Freeze3.lifted_already_GRCh37.GH.ancestry-1-5.chr22.phased.38708556-38866010.vcf.VMV1.masked.gz.best_model.vcf.gz_per_variant.tsv
```

Use --help flag for additional help:
```
python3 Compare_imputation_to_WGS.py --help
usage: Compare_imputation_to_WGS.py --ga <input_genotype_array.vcf.gz> --imputed <imputed_file.vcf.gz> --wgs <whole_genome_file.vcf.gz>
Use -h or --help to display help.


optional arguments:
  -h, --help            show this help message and exit
  --ga GA               (optional for low pass) path to genotype array file in
                        vcf.gz format, with tbi
  --wgs WGS             path to whole genome file in vcf.gz format, with tbi
  --imputed IMPUTED     path to imputed file in vcf.gz format, with tbi
  --ref REF             optional, path to reference panel file in vcf.gz
                        format, with tbi. Used for MAF calculation. WGS file
                        will be used if no reference file is provided.
  --max_total_rows MAX_TOTAL_ROWS
                        maximun number of rows or variants to be loaded
                        simultaneously, summing all chunks loaded by all cores
  --max_per_core MAX_PER_CORE
                        maximun number of variants per chunk per core, lower
                        it to avoid RAM overload
  --min_per_core MIN_PER_CORE
                        minimun number of variants per chunk per core,
                        increase to avoid interprocess communication overload
  --sout SOUT           optional output file path/name per sample, default is
                        the same as the imputed file with
                        _per_sample_results.txt suffix
  --vout VOUT           optional output file path/name per variant, default is
                        the same as the imputed file with
                        _per_variant_results.txt suffix
  --xmode XMODE         Option for developers, print additional scores.
```

### 4. Plotting results

    Adding section soon.
