# Imputation_Autoencoder
Deep learning based imputation of genetic variants, this algorithm can be used either in grid search training or in training with increasing mask ratio.

## Dependencies

- Python 3.6
- Tensorflow 1.13 (tested on 1.08 as well)

## Repository contents

- imputation_autoencoder.py: training algorithm, the actual imputation autoencoder source code
- 3_hyper_par_set.txt: an example of input hyperparameter values, containing 3 different sets of hyperparameter combinations
- input_example.tar.gz: input genotype file example

## Additional content (optional, not necessary for running the autoencoder)

- focal_loss_test_1.ipynb: an example demonstration of how the focal loss function works
- Imputation_inference_function*.ipynb: examples of how to use a pre-trained model for inference
- grid_search_summary.ipynb: summary results example for grid search

## How to run
```
python3.6 imputation_autoencoder.py input_file parameter_file save_model initial_masking_rate final_masking_rate
```

Where:
- input_file: is a input file containing genetic variants in vcf format.
- parameter_file: tab or space-delimited file containing one set of hyperparameters per line
- save_model: [True, False] whether to save a backup of the trained model in the hard disk, the saved model file name will be inference_model*
- initial_masking_rate: [float, fraction] initial masking rate, set 0 to disable masking
- final_masking_rate: [float, fraction] final masking rate,  set 0 to disable masking

A more practical example:
```
python3.6 imputation_autoencoder.py input_example.vcf 3_hyper_par_set.txt False 0.01 0.98
```
The example above will run training using input_example.vcf, applying hyperparameter values from 3_hyper_par_set.txt, not saving the model into disk (False), starting with a small masking ratio (0.01), and will keep increasing masking until it reaches 0.98 masking ratio.


## Hyperparameters

In addition to an input set of genotypes (input_example.tar.gz), the training algorithm (imputation_autoencoder.py) requires a set of hyper parameter values that are exemplified in the 3_hyper_par_set.txt file.
The hyperparameters are provided as a tab delimited list, where each column is presented in the following order:

- L1: [float] L1 (Lasso) regularizer, small values recommended, should be less than 1
- L2: [float] L2 (Ridge) regularizer, small values recommended, should be less than 1
- beta: [float] Sparsity scaling factor beta, any value grater than 1
- rho: [float] Desired average hidden layer activation (rho), less than 1
- act: [string] Activation function type, values supported: ['sigmoid','tanh', 'relu']
- LR: [float] Learning rate
- gamma: [float] scaling factor for focal loss, ignored when loss_type!=FL
- optimizer: [string] optimizer type, values supported: ['GradientDescent', 'Adam', 'RMSProp']
- loss_type: [string] loss type, values supported: ['MSE', 'CE', 'WCE', 'FL'], which respectively mean: mean squared error, cross entropy, weighted cross entropy, and focal loss.
- h_size: [float,string] hidden layer size, if 'sqrt' the hidden layer size will be equal to the square root of the input layer size, if float, the hidden layer size will be the hyperparameter value multiplied by input layer size

Example of one set of hyperparameters, following the listed order:
```
head -n 1 3_hyper_par_set.txt
1e-06 1e-06 0.001 0.07 relu 10 5 RMSProp WCE
```

Multiple sets of hyperparameter values can be provided in the same file, one set per line, for example:
```
cat 3_hyper_par_set.txt
1e-06 1e-06 0.001 0.07 relu 10 5 RMSProp WCE sqrt
1e-05 1e-08 6 0.04 tanh 1e-05 2 GradientDescent FL 1
0.01 0.0001 0.01 0.004 tanh 0.0001 0 Adam FL 0.5
```

In the example above the imputation autoencoder algorithm will train one model per hyperparameter set (3 models total).

## Results

After running the example above, the results will look like this:
TODO: adding example soon...


