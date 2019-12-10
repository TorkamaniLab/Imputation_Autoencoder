# Imputation_Autoencoder
Deep learning based imputation of genetic variants, this algorithm can be used either in grid search training or in training with increasing mask ratio.

## Dependencies

- Python 3.6
- Tensorflow 1.13 (tested on 1.08 as well)

## Configuration using pipenv

We recommend to use pipenv, which is both a package and virtual environment management tool that uses the Pipfile and Pipfile.lock files to achieve these goals (files included in this repository).
Pipenv handles the virtual environment for you in one default standard way. Below are some basics to get you started.

- If configuring a new server, please install dependencies above. Dependencies are already installed in gpomics.scripps.edu. In garibaldi gpu node, load tensorflow module:

```
module load python/3.6.3
module load tensorflow/1.8.0py36-cuda
```

- Install pipenv using pip:
```
pip install --user pipenv
```

- In the project directory (with Pipfile and Pipefile.lock), install all dependencies by pipenv. The Pipfile file provided specifies packages requirements for Python applications.
```
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
pipenv install
```

- Enter the virtual environment shell. 

```
pipenv shell
```

You are ready to run the imputation autoencoder. For more documentation on pipenv, please refer to: https://docs.pipenv.org/en/latest/

## Repository contents

- imputation_autoencoder.py: training algorithm, the actual imputation autoencoder source code
- 3_hyper_par_set.txt: an example of input hyperparameter values, containing 3 different sets of hyperparameter combinations
- input_example.tar.gz: input genotype file example

## Additional content (optional, not necessary for running the autoencoder)

- focal_loss_test_1.ipynb: an example demonstration of how the focal loss function works
- Imputation_inference_function*.ipynb: examples of how to use a pre-trained model for inference
- grid_search_summary.ipynb: summary results example for grid search
- make_hyperparameter_grid_for_grid_search.py: a tool for making grids from combinations of hyperparameter values, generating a file with the same format as 3_hyper_par_set.txt.

## How to run
```
 Usage: python3 script.py INVCF HP REC IM FM CKPT_DIR

     INVCF = [string] input file in vcf format
     HP = [string] hyperparameter list file (space delimited list of hyperparameters)
     REC = [True,False] Recovery mode, default is False
     IM = [float or fraction] Initial masking rate
     FM = [float or fraction] Final masking rate
     CKPT_DIR = [string] (optional) only needed if recovery mode is True, path to the .ckpt file
```
Example 1 (new training): 
```
python3 imputation_autoencoder.py my_input.vcf my_param.txt False 0.1 0.99
```

Example 2 (resume previous training):
```
python3 imputation_autoencoder.py my_input.vcf my_param.txt True 0.1 0.99 ./my_dir/my_model.ckpt
```

Where my_param.txt (hyperparameter file) should contain the following hyperparameters:

- L1: [float] L1 (Lasso) regularizer, small values recommended, should be less than 1, typically between 1e-2 and 1e-8
- L2: [float] L2 (Ridge) regularizer, small values recommended, should be less than 1, typically between 1e-2 and 1e-8
- beta: [float] Sparsity scaling factor beta, any value grater than 1
- rho: [float] Desired average hidden layer activation (rho), less than 1
- act: [string] Activation function type, values supported: ['sigmoid','tanh', 'relu']
- LR: [float] Learning rate
- gamma: [float] scaling factor for focal loss, ignored when loss_type!=FL
- optimizer: [string] optimizer type, values supported: ['GradientDescent', 'Adam', 'RMSProp']
- loss_type: [string] loss type, values supported: ['MSE', 'CE', 'WCE', 'FL'], which respectively mean: mean squared error, cross entropy, weighted cross entropy, and focal loss.
- h_size: [float,string] hidden layer size, if 'sqrt' the hidden layer size will be equal to the square root of the input layer size, if float, the hidden layer size will be the hyperparameter value multiplied by input layer size
- LB: [int] left buffer size, number of upstream input variants to be excluded from output layer
- RB: [int] right buffer size, number of downstream input variants to be excluded from output layer
- KP: [float vector] keep probability of each hidden layer, the number of hidden layers will be detected automatically from the number of values provided, each value must be comma separated (e.g: 1.0,0.8,0.3 for 3 hidden layers; 1.0,0.5 for 2 hidden layers, 0.5 for 1 hidden layer, etc.)

Each hyperparamter should be separated by space or tab, one hyper parameter set per line, for example (cat 3_hyper_par_set.txt):
```
#L1, L2, BETA, RHO, ACT, LR, gamma, optimizer, loss_type, h_size, LB, RB, KP
1e-06 1e-06 0.001 0.07 relu 10 5 RMSProp WCE sqrt 0 12 1
1e-05 1e-08 6 0.04 tanh 1e-05 2 GradientDescent FL 1 23 11 1,0.5
0.01 0.0001 0.01 0.004 tanh 0.0001 0 Adam FL 0.5 10 0 1,1,0.3
```

A more practical example:
```
python3.6 imputation_autoencoder.py input_example.vcf 3_hyper_par_set.txt False 0.01 0.98
```
The example above will run training using input_example.vcf, applying hyperparameter values from 3_hyper_par_set.txt, not saving the model into disk (False), starting with a small masking ratio (0.01), and will keep increasing masking until it reaches 0.98 masking ratio.
You can redirect the reports from stdout to an output file like this:
```
python3.6 imputation_autoencoder.py input_example.vcf 3_hyper_par_set.txt False 0.01 0.98 1> output.txt 2> output.log
```


In the example above the imputation autoencoder algorithm will train one model per hyperparameter set (3 models total).

## Results
The autoencoder prints detailed and summarized reports for each epoch and for the complete training process.
After running the example shown in the "How to run" session, the results will look like this:
```
grep -w "RESULT\|LABELS" output.txt
LABELS [L1, L2, BETA, RHO, ACT, LR, gamma, optimizer, loss_type, h_size, LB, RB, rsloss, rloss, sloss, acc, ac_r, ac_c, F1_micro, F1_macro, F1_weighted]
RESULT  ['1e-06', '1e-06', '0.001', '0.07', 'relu', '10', '5', 'RMSProp', 'WCE', 'sqrt', '0', '12', 2676017101.6494365, 2676017101.649409, 0.02726494355565999, 0.29184148272461796, 0.3044332385839116, 0.2272403004899808, 0.6104014711437631, 0.49455319557435573, 0.5859730799447812]
RESULT  ['1e-05', '1e-08', '6', '0.04', 'tanh', '1e-05', '2', 'GradientDescent', 'FL', '1', '23', '11', 4748104415.9408045, 4748104412.979811, 0.49349881797027567, 0.4643849319224788, 0.45409295052552784, 0.5171872712633583, 0.45847706551750844, 0.36278041530436894, 0.4258039655086927]
RESULT  ['0.01', '0.0001', '0.01', '0.004', 'tanh', '0.0001', '0', 'Adam', 'FL', '0.5', '10', '0', 182497202.90918827, 182497202.90568063, 0.3507592655291003, 0.7759625001229091, 0.8046828055396102, 0.6286148462459205, 0.7194354786524245, 0.5464695094504292, 0.8942798322320101]
```

Where the first 10 values are just the hyperparameters set by the user (L1, L2, BETA, RHO, ACT, LR, gamma, optimizer, loss_type, h_size), the next columns represent the resulting performance metrics for their respective hyperparameter set:

- rsloss: final loss result combined with sparsity penalty (depends on loss type selected by the user, beta and rho)
- rloss: reconstruction loss without sparsity penalty (depends on loss type selected by the user)
- sloss: sparsity loss (depends on beta and rho)
- acc: accuracy (proportion of correct predictions versus total number of predictions)
- ac_r: accuracy for MAF <= 0.01 
- ac_c: accuracy for MAF > 0.01
- F1_micro: F1 score (micro, per sample)
- F1_macro: F1 score (macro, per feature)
- F1_weighted: F1 score (weighted, typical F1-score across all genotypes)

The summary results are shown in one 'RESULT  [*]' line per hyperparameter set.

## Making a new hyperparameter grid
You can change your list of possible hyperparameter values by editing the python script named make_hyperparameter_grid_for_grid_search.py.
This is the current list of the hyper parameters set (copied from make_hyperparameter_grid_for_grid_search.py), which can be edited to either include new values or exclude old ones:

```
act_arr = ['tanh']
l1_arr = [1e-3,1e-4,1e-5,1e-6,1e-1,1e-2,1e-7,1e-8]
l2_arr = [0]
beta_arr = [1,2,4,6,8,10]
rho_arr = [0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1.0]
learning_rate_arr = [0.00001, 0.0001, 0.001]
gamma_arr = [0,0.5,1,2,3,5]
opt_array = ["RMSProp"]
loss_arr = ["FL"]
hs_arr = ['sqrt', '0.10', '0.20', '0.40', '0.60', '0.80', '1']
lb_arr = [0]
rb_arr = [0]
```

Once you are satisfied with the list of possible values for each hyperparameter, then you can make a new grid:
```
python3.6 make_hyperparameter_grid_for_grid_search.py
```

If it works, the script will print the following report to stdout:
```
Building grid search combinations.
Extracted 100 from 60480 possible combinations.
Saving new grid to hyper_parameter_list.100.txt
Building grid search combinations.
Extracted 500 from 60480 possible combinations.
Saving new grid to hyper_parameter_list.500.txt
Building grid search combinations.
Extracted 1000 from 60480 possible combinations.
Saving new grid to hyper_parameter_list.1000.txt
Building grid search combinations.
Extracted 5000 from 60480 possible combinations.
Saving new grid to hyper_parameter_list.5000.txt
Building grid search combinations.
Extracted 10000 from 60480 possible combinations.
Saving new grid to hyper_parameter_list.10000.txt
```

A set of multiple hyper_parameter_list.N.txt files is generated, where N is the number of random combinations of hyperparameter values extracted from the full grid (60480 possible combinations in this example).
If you list the contents of one of these files, it should look like this:
```
head hyper_parameter_list.10000.txt
0.01 0 8 0.07 tanh 0.0001 0 RMSProp FL 0.10 0 0
1e-06 0 8 0.1 tanh 0.001 2 RMSProp FL sqrt 0 0
0.01 0 8 0.01 tanh 0.0001 0 RMSProp FL 0.20 0 0
1e-08 0 6 0.04 tanh 1e-05 3 RMSProp FL sqrt 0 0
1e-08 0 4 0.007 tanh 0.0001 0.5 RMSProp FL sqrt 0 0
1e-07 0 6 0.007 tanh 1e-05 5 RMSProp FL sqrt 0 0
1e-08 0 8 0.01 tanh 1e-05 0.5 RMSProp FL sqrt 0 0
0.0001 0 4 0.007 tanh 0.0001 0 RMSProp FL 0.20 0 0
0.0001 0 4 0.01 tanh 0.0001 0.5 RMSProp FL 0.60 0 0
1e-05 0 10 0.01 tanh 0.0001 1 RMSProp FL 0.80 0 0
```


## Running Inference

Once you ran training, a pre-trained model will be saved into a tensorflow checkpoint file and meta graph.
For running inference using the pre-trained model and a new input file:

```
python3 Imputation_inference_function_only_with_dosage.py

Usage: python3 Imputation_inference_function_only_with_dosage.py reference.1-5 genotype_array model_file output_file_name

       reference.1-5:       first 5 columns of reference panel VCF file (chromosome position rsID REF ALT), used to build imputed file
       genotype_array:      genotype array file in VCF format, file to be imputed
       model_file:          pretrained model directory path (just directory path, no file name and no extension required
       output_file_name:    (optional) a name for the output file, imputed file in VCF format, same name prefix as input if no out name is provided
```


For example, using HRC dataset:

```
python3 Imputation_inference_function_only_with_dosage.py HRC.r1-1.EGA.GRCh37.chr22.haplotypes.20227467-20247236.vcf.VMV1.024.1-5 HRC.r1-1.EGA.GRCh37.chr22.haplotypes.20227467-20247236.vcf.VMV1.024.masked HRC.r1-1.EGA.GRCh37.chr22.haplotypes.20227467-20247236.vcf.VMV1_model HRC.r1-1.EGA.GRCh37.chr22.haplotypes.20227467-20247236.vcf.VMV1.024.masked.autoencoder_imputed_round1.vcf

```

The standard output message should look like:

```
Time to do inference (sec):  0.27009798999642953
RESULT: HRC.r1-1.EGA.GRCh37.chr22.haplotypes.20227467-20247236.vcf.VMV1/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.20227467-20247236.vcf.VMV1.024.masked.autoencoder_imputed_round1.vcf
Time to write output file (sec):  0.03913883300265297

```

Where the *.autoencoder_imputed_round1.vcf file is the imputed result in VCF format:

```
##fileformat=VCFv4.1
##filedate=2019-10-30
##source=Imputation_autoencoder
##contig=<ID=22>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DS,Number=1,Type=Float,Description="Estimated Alternate Allele Dosage : [P(0/1)+2*P(1/1)]">
##FORMAT=<ID=Paa,Number=1,Type=Float,Description="Imputation probability for homozigous reference : Pa=y_pred[i][j]*(1-y_pred[i][j+1])">
##FORMAT=<ID=Pab,Number=1,Type=Float,Description="Imputation probability for heterozigous : Pab=y_pred[i][j]*y_pred[i][j+1]">
##FORMAT=<ID=Pbb,Number=1,Type=Float,Description="Imputation probability for homozigous alternative : Pb=(1-y_pred[i][j])*y_pred[i][j+1]">
##FORMAT=<ID=AP,Number=1,Type=Float,Description="Predicted presence of reference allele (autoencoder raw output)">
##FORMAT=<ID=BP,Number=1,Type=Float,Description="Predicted presence of alternative allele (autoencoder raw output)">
#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT  1KGTEST
22      20227551        rs701430        A       G       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 1/1:1.9521:2.1671086955780534e-10:1.9123217699738734e-18:8.824300122031762e-09:2.1671087147012713e-10:8.824300123944084e-09
22      20227563        rs114074551     A       G       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20227595        rs73877201      G       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20227689        rs573952894     G       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20227958        rs191917535     A       G       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20227971        rs530513657     C       T       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228055        rs149600212     G       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228195        rs117150443     G       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228238        rs567146400     G       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228365        rs559206462     G       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228388        rs60421391      G       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228465        rs701429        G       T       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/1:1.0:0.0:1.0:0.0:1.0:1.0
22      20228483        .       C       T       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228486        rs58660506      C       T       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228487        rs887764        G       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0:1:0:0:1:0
22      20228542        rs701428        A       G       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 1/1:2.0:0.0:0.0:1.0:0.0:1.0
22      20228557        rs115516414     C       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228594        rs114995292     G       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228616        rs546557732     C       T       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228668        rs182926867     C       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228698        rs142280269     G       C       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/0:0.0:1.0:0.0:0.0:1.0:0.0
22      20228764        rs34031696      G       A       .       .       .       GT:DS:Paa:Pab:Pbb:AP:BP 0/1:1.0:1.840345653647546e-10:0.9999999998159654:0.0:1.0:0.9999999998159654

```
