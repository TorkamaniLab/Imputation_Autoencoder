CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_1 --l1 1e-07 --l1 1e-08 --beta 0.0001 --rho 0.05 --gamma 0.0 --disable_alpha 1 --learn_rate 0.001 --activation leakyrelu --optimizer rmsprop --loss_type FL --n_layers 8 --size_ratio 0.7 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_2 --l1 0.0 --l1 0.01 --beta 0.0 --rho 0.5 --gamma 4.0 --disable_alpha 1 --learn_rate 0.01 --activation leakyrelu --optimizer rmsprop --loss_type CE --n_layers 6 --size_ratio 0.9 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_3 --l1 0.1 --l1 1e-07 --beta 0.1 --rho 0.5 --gamma 1.0 --disable_alpha 0 --learn_rate 0.1 --activation tanh --optimizer rmsprop --loss_type CE --n_layers 8 --size_ratio 1.0 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_4 --l1 1e-05 --l1 0.01 --beta 1.0 --rho 0.5 --gamma 5.0 --disable_alpha 1 --learn_rate 1e-05 --activation leakyrelu --optimizer adadelta --loss_type CE --n_layers 8 --size_ratio 0.8 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_5 --l1 1e-07 --l1 1e-07 --beta 0.0 --rho 0.005 --gamma 0.5 --disable_alpha 0 --learn_rate 1e-05 --activation leakyrelu --optimizer adagrad --loss_type FL --n_layers 2 --size_ratio 0.9 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_6 --l1 1e-05 --l1 0.001 --beta 0.0001 --rho 0.05 --gamma 1.0 --disable_alpha 0 --learn_rate 0.001 --activation sigmoid --optimizer rmsprop --loss_type CE --n_layers 4 --size_ratio 0.9 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_7 --l1 0.0 --l1 1e-08 --beta 0.0 --rho 0.5 --gamma 0.5 --disable_alpha 0 --learn_rate 1e-05 --activation leakyrelu --optimizer sgd --loss_type FL --n_layers 4 --size_ratio 0.6 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_8 --l1 0.0 --l1 0.0 --beta 1.0 --rho 0.05 --gamma 1.0 --disable_alpha 0 --learn_rate 0.1 --activation tanh --optimizer rmsprop --loss_type FL --n_layers 4 --size_ratio 0.9 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_9 --l1 1e-05 --l1 1e-07 --beta 0.001 --rho 0.5 --gamma 1.0 --disable_alpha 1 --learn_rate 1e-05 --activation relu --optimizer adam --loss_type FL --n_layers 6 --size_ratio 0.2 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_10 --l1 1e-07 --l1 0.01 --beta 5.0 --rho 0.01 --gamma 1.0 --disable_alpha 1 --learn_rate 0.1 --activation leakyrelu --optimizer rmsprop --loss_type CE --n_layers 4 --size_ratio 1.0 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_11 --l1 0.0001 --l1 0.0 --beta 0.0001 --rho 0.5 --gamma 5.0 --disable_alpha 0 --learn_rate 0.001 --activation tanh --optimizer adadelta --loss_type FL --n_layers 6 --size_ratio 0.6 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_12 --l1 1e-06 --l1 0.0 --beta 0.01 --rho 0.05 --gamma 4.0 --disable_alpha 0 --learn_rate 0.0001 --activation relu --optimizer adam --loss_type FL --n_layers 4 --size_ratio 0.9 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_13 --l1 0.0 --l1 1e-07 --beta 0.0001 --rho 0.005 --gamma 3.0 --disable_alpha 1 --learn_rate 0.01 --activation sigmoid --optimizer adagrad --loss_type FL --n_layers 8 --size_ratio 0.4 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_14 --l1 0.001 --l1 1e-07 --beta 1.0 --rho 0.01 --gamma 2.0 --disable_alpha 0 --learn_rate 0.0001 --activation tanh --optimizer adagrad --loss_type CE --n_layers 2 --size_ratio 0.5 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_15 --l1 0.0 --l1 1e-08 --beta 0.01 --rho 0.25 --gamma 5.0 --disable_alpha 0 --learn_rate 1e-05 --activation leakyrelu --optimizer sgd --loss_type FL --n_layers 2 --size_ratio 0.7 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_16 --l1 0.01 --l1 1e-05 --beta 0.0 --rho 0.5 --gamma 0.5 --disable_alpha 1 --learn_rate 0.001 --activation sigmoid --optimizer adagrad --loss_type FL --n_layers 6 --size_ratio 0.2 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_17 --l1 1e-05 --l1 1e-05 --beta 0.0 --rho 0.01 --gamma 1.0 --disable_alpha 1 --learn_rate 1e-05 --activation leakyrelu --optimizer sgd --loss_type FL --n_layers 4 --size_ratio 0.8 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_18 --l1 0.001 --l1 1e-07 --beta 0.1 --rho 0.1 --gamma 0.0 --disable_alpha 1 --learn_rate 0.001 --activation relu --optimizer sgd --loss_type FL --n_layers 2 --size_ratio 0.2 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_19 --l1 1e-08 --l1 0.0001 --beta 0.0001 --rho 0.25 --gamma 4.0 --disable_alpha 0 --learn_rate 0.1 --activation leakyrelu --optimizer adagrad --loss_type FL --n_layers 2 --size_ratio 0.2 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_20 --l1 0.001 --l1 1e-05 --beta 5.0 --rho 0.5 --gamma 3.0 --disable_alpha 0 --learn_rate 0.01 --activation relu --optimizer rmsprop --loss_type CE --n_layers 6 --size_ratio 0.3 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_21 --l1 0.0 --l1 0.1 --beta 1.0 --rho 0.005 --gamma 5.0 --disable_alpha 0 --learn_rate 0.001 --activation sigmoid --optimizer rmsprop --loss_type CE --n_layers 2 --size_ratio 0.2 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_22 --l1 1e-05 --l1 0.01 --beta 0.0 --rho 0.05 --gamma 3.0 --disable_alpha 1 --learn_rate 1e-05 --activation relu --optimizer sgd --loss_type FL --n_layers 4 --size_ratio 0.8 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_23 --l1 1e-07 --l1 0.001 --beta 0.01 --rho 0.5 --gamma 4.0 --disable_alpha 0 --learn_rate 0.001 --activation tanh --optimizer adadelta --loss_type FL --n_layers 8 --size_ratio 0.2 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_24 --l1 1e-08 --l1 0.001 --beta 5.0 --rho 0.25 --gamma 5.0 --disable_alpha 1 --learn_rate 0.0001 --activation sigmoid --optimizer adagrad --loss_type FL --n_layers 6 --size_ratio 0.8 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_25 --l1 1e-07 --l1 0.1 --beta 0.0001 --rho 0.25 --gamma 0.0 --disable_alpha 1 --learn_rate 0.001 --activation sigmoid --optimizer sgd --loss_type CE --n_layers 6 --size_ratio 0.2 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_26 --l1 0.1 --l1 1e-05 --beta 0.1 --rho 0.5 --gamma 2.0 --disable_alpha 0 --learn_rate 0.0001 --activation leakyrelu --optimizer sgd --loss_type FL --n_layers 8 --size_ratio 0.2 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_27 --l1 1e-06 --l1 0.0 --beta 0.01 --rho 0.1 --gamma 2.0 --disable_alpha 0 --learn_rate 1e-05 --activation relu --optimizer adagrad --loss_type CE --n_layers 2 --size_ratio 0.9 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_28 --l1 1e-05 --l1 0.0 --beta 1e-05 --rho 0.5 --gamma 1.0 --disable_alpha 0 --learn_rate 0.1 --activation tanh --optimizer rmsprop --loss_type CE --n_layers 2 --size_ratio 0.2 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_29 --l1 1e-07 --l1 0.001 --beta 5.0 --rho 0.5 --gamma 2.0 --disable_alpha 0 --learn_rate 0.01 --activation leakyrelu --optimizer adam --loss_type FL --n_layers 6 --size_ratio 0.7 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_30 --l1 0.001 --l1 0.1 --beta 0.0001 --rho 0.5 --gamma 1.0 --disable_alpha 0 --learn_rate 1e-05 --activation sigmoid --optimizer rmsprop --loss_type CE --n_layers 2 --size_ratio 0.7 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_31 --l1 1e-08 --l1 1e-06 --beta 0.001 --rho 0.005 --gamma 3.0 --disable_alpha 0 --learn_rate 0.01 --activation relu --optimizer adam --loss_type CE --n_layers 8 --size_ratio 0.8 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_32 --l1 1e-06 --l1 1e-08 --beta 0.001 --rho 0.01 --gamma 0.5 --disable_alpha 1 --learn_rate 0.0001 --activation leakyrelu --optimizer rmsprop --loss_type FL --n_layers 4 --size_ratio 1.0 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_33 --l1 0.1 --l1 0.01 --beta 1e-05 --rho 0.1 --gamma 0.5 --disable_alpha 0 --learn_rate 0.0001 --activation sigmoid --optimizer adam --loss_type CE --n_layers 4 --size_ratio 0.4 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_34 --l1 1e-07 --l1 0.0001 --beta 1.0 --rho 0.1 --gamma 1.0 --disable_alpha 0 --learn_rate 0.1 --activation tanh --optimizer sgd --loss_type CE --n_layers 4 --size_ratio 0.3 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_35 --l1 0.001 --l1 0.0001 --beta 0.1 --rho 0.05 --gamma 5.0 --disable_alpha 1 --learn_rate 1e-05 --activation leakyrelu --optimizer adagrad --loss_type CE --n_layers 6 --size_ratio 0.7 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_36 --l1 1e-07 --l1 1e-08 --beta 5.0 --rho 0.05 --gamma 0.5 --disable_alpha 1 --learn_rate 1e-05 --activation sigmoid --optimizer sgd --loss_type FL --n_layers 4 --size_ratio 0.6 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_37 --l1 1e-07 --l1 1e-05 --beta 0.1 --rho 0.01 --gamma 0.5 --disable_alpha 1 --learn_rate 1e-05 --activation tanh --optimizer adadelta --loss_type CE --n_layers 8 --size_ratio 0.6 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_38 --l1 1e-06 --l1 0.001 --beta 0.1 --rho 0.05 --gamma 3.0 --disable_alpha 1 --learn_rate 0.1 --activation relu --optimizer rmsprop --loss_type FL --n_layers 6 --size_ratio 1.0 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_39 --l1 0.0001 --l1 0.1 --beta 1.0 --rho 0.05 --gamma 3.0 --disable_alpha 1 --learn_rate 0.0001 --activation leakyrelu --optimizer rmsprop --loss_type FL --n_layers 8 --size_ratio 0.4 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_40 --l1 1e-05 --l1 1e-08 --beta 0.001 --rho 0.1 --gamma 5.0 --disable_alpha 1 --learn_rate 0.001 --activation tanh --optimizer adagrad --loss_type FL --n_layers 6 --size_ratio 0.5 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_41 --l1 1e-08 --l1 1e-08 --beta 0.0001 --rho 0.05 --gamma 1.0 --disable_alpha 0 --learn_rate 0.001 --activation relu --optimizer adadelta --loss_type FL --n_layers 4 --size_ratio 0.8 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_42 --l1 0.0 --l1 1e-05 --beta 0.0001 --rho 0.05 --gamma 0.5 --disable_alpha 1 --learn_rate 0.001 --activation tanh --optimizer rmsprop --loss_type CE --n_layers 2 --size_ratio 0.3 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_43 --l1 1e-08 --l1 0.0 --beta 1e-05 --rho 0.5 --gamma 3.0 --disable_alpha 0 --learn_rate 1e-05 --activation sigmoid --optimizer adagrad --loss_type CE --n_layers 2 --size_ratio 0.8 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_44 --l1 0.0001 --l1 1e-05 --beta 0.0001 --rho 0.25 --gamma 0.5 --disable_alpha 1 --learn_rate 0.0001 --activation sigmoid --optimizer sgd --loss_type FL --n_layers 6 --size_ratio 0.7 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_45 --l1 0.0 --l1 0.01 --beta 0.01 --rho 0.05 --gamma 4.0 --disable_alpha 1 --learn_rate 0.1 --activation relu --optimizer rmsprop --loss_type FL --n_layers 4 --size_ratio 1.0 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_46 --l1 1e-05 --l1 0.01 --beta 0.0001 --rho 0.005 --gamma 4.0 --disable_alpha 0 --learn_rate 0.0001 --activation leakyrelu --optimizer adam --loss_type CE --n_layers 8 --size_ratio 0.3 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_47 --l1 1e-05 --l1 0.1 --beta 0.001 --rho 0.25 --gamma 1.0 --disable_alpha 0 --learn_rate 0.0001 --activation sigmoid --optimizer adam --loss_type CE --n_layers 2 --size_ratio 0.8 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_48 --l1 0.001 --l1 1e-07 --beta 1e-05 --rho 0.25 --gamma 4.0 --disable_alpha 1 --learn_rate 1e-05 --activation tanh --optimizer adadelta --loss_type FL --n_layers 2 --size_ratio 1.0 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_49 --l1 1e-06 --l1 1e-08 --beta 0.0001 --rho 0.005 --gamma 0.0 --disable_alpha 0 --learn_rate 0.001 --activation tanh --optimizer rmsprop --loss_type FL --n_layers 8 --size_ratio 0.6 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_50 --l1 0.1 --l1 1e-08 --beta 0.001 --rho 0.005 --gamma 3.0 --disable_alpha 0 --learn_rate 0.001 --activation sigmoid --optimizer adam --loss_type CE --n_layers 8 --size_ratio 1.0 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_51 --l1 1e-06 --l1 0.001 --beta 0.01 --rho 0.5 --gamma 0.0 --disable_alpha 0 --learn_rate 0.1 --activation tanh --optimizer adagrad --loss_type CE --n_layers 4 --size_ratio 0.3 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_52 --l1 0.001 --l1 0.0 --beta 0.1 --rho 0.5 --gamma 0.0 --disable_alpha 1 --learn_rate 0.1 --activation relu --optimizer rmsprop --loss_type FL --n_layers 8 --size_ratio 0.7 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_53 --l1 1e-07 --l1 1e-06 --beta 0.001 --rho 0.01 --gamma 3.0 --disable_alpha 1 --learn_rate 1e-05 --activation tanh --optimizer adadelta --loss_type FL --n_layers 8 --size_ratio 0.8 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_54 --l1 0.0001 --l1 1e-08 --beta 0.0001 --rho 0.1 --gamma 0.5 --disable_alpha 0 --learn_rate 0.1 --activation tanh --optimizer rmsprop --loss_type CE --n_layers 2 --size_ratio 0.3 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_55 --l1 1e-06 --l1 0.0001 --beta 1.0 --rho 0.1 --gamma 2.0 --disable_alpha 1 --learn_rate 0.001 --activation sigmoid --optimizer adagrad --loss_type CE --n_layers 8 --size_ratio 0.5 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_56 --l1 1e-06 --l1 0.1 --beta 1e-05 --rho 0.005 --gamma 3.0 --disable_alpha 1 --learn_rate 1e-05 --activation sigmoid --optimizer adadelta --loss_type CE --n_layers 2 --size_ratio 0.2 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_57 --l1 1e-08 --l1 1e-05 --beta 0.01 --rho 0.25 --gamma 5.0 --disable_alpha 1 --learn_rate 0.001 --activation leakyrelu --optimizer rmsprop --loss_type FL --n_layers 2 --size_ratio 1.0 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_58 --l1 1e-06 --l1 1e-07 --beta 0.001 --rho 0.1 --gamma 1.0 --disable_alpha 1 --learn_rate 0.01 --activation tanh --optimizer adagrad --loss_type FL --n_layers 6 --size_ratio 0.3 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_59 --l1 1e-08 --l1 0.1 --beta 0.0001 --rho 0.05 --gamma 2.0 --disable_alpha 0 --learn_rate 0.01 --activation sigmoid --optimizer sgd --loss_type CE --n_layers 2 --size_ratio 1.0 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_60 --l1 0.0 --l1 0.1 --beta 0.01 --rho 0.1 --gamma 4.0 --disable_alpha 1 --learn_rate 0.001 --activation leakyrelu --optimizer sgd --loss_type FL --n_layers 4 --size_ratio 0.6 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_61 --l1 1e-07 --l1 0.001 --beta 1.0 --rho 0.1 --gamma 4.0 --disable_alpha 1 --learn_rate 1e-05 --activation sigmoid --optimizer rmsprop --loss_type FL --n_layers 6 --size_ratio 0.6 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_62 --l1 1e-07 --l1 0.0 --beta 0.0001 --rho 0.01 --gamma 4.0 --disable_alpha 0 --learn_rate 0.01 --activation sigmoid --optimizer adadelta --loss_type CE --n_layers 6 --size_ratio 1.0 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_63 --l1 0.0001 --l1 0.1 --beta 0.01 --rho 0.05 --gamma 3.0 --disable_alpha 0 --learn_rate 0.01 --activation leakyrelu --optimizer adagrad --loss_type FL --n_layers 6 --size_ratio 1.0 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_64 --l1 0.0 --l1 0.1 --beta 0.1 --rho 0.005 --gamma 1.0 --disable_alpha 0 --learn_rate 0.1 --activation leakyrelu --optimizer sgd --loss_type CE --n_layers 4 --size_ratio 0.3 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_65 --l1 0.001 --l1 0.0 --beta 0.0001 --rho 0.05 --gamma 2.0 --disable_alpha 0 --learn_rate 0.1 --activation tanh --optimizer adadelta --loss_type CE --n_layers 6 --size_ratio 0.9 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_66 --l1 0.1 --l1 0.0 --beta 0.0 --rho 0.25 --gamma 0.0 --disable_alpha 0 --learn_rate 1e-05 --activation relu --optimizer adadelta --loss_type CE --n_layers 4 --size_ratio 0.4 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_67 --l1 0.1 --l1 1e-08 --beta 0.01 --rho 0.05 --gamma 4.0 --disable_alpha 1 --learn_rate 0.0001 --activation leakyrelu --optimizer rmsprop --loss_type FL --n_layers 8 --size_ratio 0.4 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_68 --l1 0.01 --l1 0.1 --beta 1.0 --rho 0.005 --gamma 5.0 --disable_alpha 1 --learn_rate 0.1 --activation leakyrelu --optimizer rmsprop --loss_type CE --n_layers 2 --size_ratio 0.5 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_69 --l1 1e-06 --l1 0.0001 --beta 0.1 --rho 0.5 --gamma 5.0 --disable_alpha 0 --learn_rate 0.0001 --activation sigmoid --optimizer adam --loss_type FL --n_layers 8 --size_ratio 0.4 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_70 --l1 0.0001 --l1 1e-07 --beta 0.0001 --rho 0.005 --gamma 5.0 --disable_alpha 1 --learn_rate 0.0001 --activation leakyrelu --optimizer adadelta --loss_type FL --n_layers 2 --size_ratio 0.9 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_71 --l1 1e-06 --l1 1e-05 --beta 0.1 --rho 0.1 --gamma 4.0 --disable_alpha 0 --learn_rate 0.0001 --activation relu --optimizer rmsprop --loss_type FL --n_layers 8 --size_ratio 0.8 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_72 --l1 0.1 --l1 0.001 --beta 0.1 --rho 0.005 --gamma 3.0 --disable_alpha 0 --learn_rate 0.0001 --activation relu --optimizer sgd --loss_type FL --n_layers 4 --size_ratio 0.8 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_73 --l1 0.1 --l1 0.0 --beta 0.0001 --rho 0.05 --gamma 0.5 --disable_alpha 0 --learn_rate 0.001 --activation relu --optimizer sgd --loss_type CE --n_layers 6 --size_ratio 0.9 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_74 --l1 1e-08 --l1 1e-05 --beta 0.0 --rho 0.25 --gamma 3.0 --disable_alpha 0 --learn_rate 1e-05 --activation sigmoid --optimizer adagrad --loss_type FL --n_layers 8 --size_ratio 1.0 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_75 --l1 1e-07 --l1 0.001 --beta 0.0001 --rho 0.005 --gamma 5.0 --disable_alpha 0 --learn_rate 0.1 --activation leakyrelu --optimizer adadelta --loss_type FL --n_layers 8 --size_ratio 0.9 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_76 --l1 1e-07 --l1 0.01 --beta 0.001 --rho 0.05 --gamma 2.0 --disable_alpha 1 --learn_rate 0.001 --activation sigmoid --optimizer adadelta --loss_type FL --n_layers 2 --size_ratio 1.0 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_77 --l1 1e-06 --l1 0.0 --beta 0.0001 --rho 0.01 --gamma 1.0 --disable_alpha 0 --learn_rate 0.01 --activation leakyrelu --optimizer rmsprop --loss_type CE --n_layers 8 --size_ratio 0.8 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_78 --l1 1e-05 --l1 1e-06 --beta 0.01 --rho 0.25 --gamma 2.0 --disable_alpha 0 --learn_rate 0.0001 --activation leakyrelu --optimizer adadelta --loss_type CE --n_layers 4 --size_ratio 0.3 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_79 --l1 1e-07 --l1 0.001 --beta 5.0 --rho 0.1 --gamma 1.0 --disable_alpha 1 --learn_rate 0.0001 --activation leakyrelu --optimizer adadelta --loss_type CE --n_layers 2 --size_ratio 0.4 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_80 --l1 1e-08 --l1 1e-07 --beta 0.01 --rho 0.01 --gamma 3.0 --disable_alpha 1 --learn_rate 0.001 --activation tanh --optimizer rmsprop --loss_type CE --n_layers 8 --size_ratio 0.7 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_81 --l1 1e-06 --l1 1e-08 --beta 0.1 --rho 0.05 --gamma 4.0 --disable_alpha 0 --learn_rate 0.0001 --activation sigmoid --optimizer adagrad --loss_type FL --n_layers 4 --size_ratio 0.7 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_82 --l1 0.001 --l1 0.001 --beta 1e-05 --rho 0.1 --gamma 0.5 --disable_alpha 0 --learn_rate 1e-05 --activation sigmoid --optimizer adagrad --loss_type CE --n_layers 2 --size_ratio 0.8 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_83 --l1 1e-06 --l1 0.001 --beta 0.0001 --rho 0.25 --gamma 3.0 --disable_alpha 1 --learn_rate 1e-05 --activation sigmoid --optimizer adadelta --loss_type FL --n_layers 8 --size_ratio 1.0 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_84 --l1 0.1 --l1 0.0 --beta 0.01 --rho 0.01 --gamma 0.0 --disable_alpha 1 --learn_rate 0.0001 --activation tanh --optimizer adam --loss_type CE --n_layers 4 --size_ratio 1.0 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_85 --l1 0.1 --l1 0.01 --beta 5.0 --rho 0.05 --gamma 5.0 --disable_alpha 1 --learn_rate 0.0001 --activation tanh --optimizer rmsprop --loss_type CE --n_layers 8 --size_ratio 0.2 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_86 --l1 1e-08 --l1 0.001 --beta 0.001 --rho 0.05 --gamma 5.0 --disable_alpha 0 --learn_rate 1e-05 --activation leakyrelu --optimizer sgd --loss_type CE --n_layers 4 --size_ratio 0.8 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_87 --l1 1e-06 --l1 1e-06 --beta 0.1 --rho 0.01 --gamma 0.0 --disable_alpha 0 --learn_rate 0.1 --activation tanh --optimizer adam --loss_type CE --n_layers 6 --size_ratio 0.4 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_88 --l1 0.1 --l1 1e-08 --beta 0.001 --rho 0.005 --gamma 0.0 --disable_alpha 1 --learn_rate 0.1 --activation sigmoid --optimizer adagrad --loss_type CE --n_layers 4 --size_ratio 0.3 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_89 --l1 1e-05 --l1 0.0001 --beta 1.0 --rho 0.25 --gamma 1.0 --disable_alpha 0 --learn_rate 0.001 --activation leakyrelu --optimizer adam --loss_type FL --n_layers 6 --size_ratio 0.8 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_90 --l1 1e-08 --l1 0.001 --beta 1e-05 --rho 0.5 --gamma 3.0 --disable_alpha 0 --learn_rate 0.01 --activation leakyrelu --optimizer adadelta --loss_type CE --n_layers 4 --size_ratio 0.7 --decay_rate 0.95
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_91 --l1 0.1 --l1 0.001 --beta 0.1 --rho 0.25 --gamma 3.0 --disable_alpha 1 --learn_rate 0.1 --activation tanh --optimizer sgd --loss_type FL --n_layers 8 --size_ratio 0.7 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_92 --l1 0.0001 --l1 1e-08 --beta 0.1 --rho 0.25 --gamma 4.0 --disable_alpha 1 --learn_rate 0.0001 --activation sigmoid --optimizer adagrad --loss_type CE --n_layers 4 --size_ratio 0.7 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_93 --l1 0.0 --l1 1e-08 --beta 0.01 --rho 0.25 --gamma 1.0 --disable_alpha 1 --learn_rate 0.0001 --activation relu --optimizer sgd --loss_type CE --n_layers 8 --size_ratio 0.8 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_94 --l1 0.0 --l1 0.0001 --beta 1e-05 --rho 0.1 --gamma 3.0 --disable_alpha 1 --learn_rate 0.001 --activation tanh --optimizer adam --loss_type CE --n_layers 2 --size_ratio 1.0 --decay_rate 0.25
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_95 --l1 1e-05 --l1 1e-05 --beta 1.0 --rho 0.25 --gamma 0.0 --disable_alpha 1 --learn_rate 0.0001 --activation relu --optimizer adadelta --loss_type FL --n_layers 8 --size_ratio 0.2 --decay_rate 0.5
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_96 --l1 0.001 --l1 0.01 --beta 0.01 --rho 0.05 --gamma 0.5 --disable_alpha 1 --learn_rate 1e-05 --activation leakyrelu --optimizer adadelta --loss_type CE --n_layers 2 --size_ratio 0.8 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=0 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_97 --l1 0.1 --l1 0.001 --beta 5.0 --rho 0.01 --gamma 4.0 --disable_alpha 1 --learn_rate 0.001 --activation relu --optimizer sgd --loss_type FL --n_layers 2 --size_ratio 0.3 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=1 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_98 --l1 1e-08 --l1 0.01 --beta 0.0 --rho 0.5 --gamma 0.5 --disable_alpha 1 --learn_rate 0.001 --activation tanh --optimizer sgd --loss_type CE --n_layers 4 --size_ratio 0.8 --decay_rate 0.0
CUDA_VISIBLE_DEVICES=2 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_99 --l1 0.0 --l1 0.0 --beta 0.0 --rho 0.25 --gamma 4.0 --disable_alpha 0 --learn_rate 0.1 --activation leakyrelu --optimizer adadelta --loss_type FL --n_layers 4 --size_ratio 0.2 --decay_rate 0.75
CUDA_VISIBLE_DEVICES=3 python3 DSAE_TORCH_ARG.py --input examples/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.38708556-38866010.vcf.VMV1 --min_mask 0.80 --max_mask 0.99755620723362658846 --model_id model_100 --l1 1e-05 --l1 0.01 --beta 5.0 --rho 0.05 --gamma 3.0 --disable_alpha 1 --learn_rate 0.0001 --activation tanh --optimizer rmsprop --loss_type CE --n_layers 4 --size_ratio 0.9 --decay_rate 0.95
