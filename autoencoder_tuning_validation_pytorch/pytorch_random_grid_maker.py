
# coding: utf-8

# In[78]:


import numpy as np 
import csv
import sys

#use 10 just for debugging, increase this number on final run
sample_size = 10
sample_size = int(sys.argv[1])

starting_id = int(sys.argv[2])

#activation
act_arr = ['relu', 'tanh', 'sigmoid', 'leakyrelu']
#L1
l1_arr = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8, 0.]
#L2
l2_arr = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8, 0.]
#BETA
beta_arr = [0,1e-1,1e-2,1e-3,1e-4,1e-5,1,5]
#RHO
rho_arr = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
#learning rate
learning_rate_arr = [1e-1,1e-2,1e-3,1e-4,1e-5]
#GAMMA
gamma_arr = [0,0.5,1,2,3,4,5]
#optimizer
opt_array = ["rmsprop", "sgd", "adam", "adadelta", "adagrad"]
#loss type
loss_arr = ["FL", "CE"]
#hidden layer size ratio (will be multiplied by the output feature size)
hs_arr = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.20]
#learning rate decay rate, 0=disabled decay
decay_arr = [0.0, 0.25, 0.5, 0.75, 0.95]
#disable/enable alpha in focal loss (FL)
alpha_arr = [0,1]
#total number of layers (half encoding, half decoding)
nl_arr = [2,4,6,8]

hp_list = [l1_arr, l2_arr, beta_arr, rho_arr, gamma_arr, alpha_arr,
           learning_rate_arr, act_arr, opt_array, loss_arr, nl_arr,
           hs_arr, decay_arr]

hp_labels = ['model_id','l1', 'l2', 'beta', 'rho', 'gamma', 'disable_alpha',
             'learn_rate', 'activation', 'optimizer', 'loss_type', 'n_layers',
             'size_ratio', 'decay_rate']

comment_line = hp_labels[:] #independent copy
comment_line[0] = "#"+comment_line[0]
sampled_hps = []

script_name="DSAE_TORCH_ARG.py"

for i in range(sample_size):
    #hp_set = ['model_'+str(i+1)]    
    hp_set = ['model_'+str(i+starting_id)]    
    for hp_sublist in hp_list:
        hp_set += [np.random.choice(hp_sublist)]
    #print(hp_set)
    sampled_hps  += [hp_set]

with open(str(sample_size)+'_random_hyperparameters.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    tsv_output.writerow(comment_line)

print(hp_labels)

with open(str(sample_size)+'_random_hyperparameters.tsv', 'a+', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    
    for hps in sampled_hps:
        print(hps)
        tsv_output.writerow(hps)


# In[79]:


with open(str(sample_size)+'_random_hyperparameters.sh', 'w') as file_object:
    
    for hps in sampled_hps:
        #print(hps)
        cmd="CUDA_VISIBLE_DEVICES=<my_GPU_id> python3 "+script_name+" --input <my_input_file> --min_mask <my_min_mask> --max_mask <my_max_mask>"
        for label, value in zip(hp_labels, hps):
            cmd+=" --"+label+" "+str(value)
        print(cmd+" --resume 1")
        file_object.write(cmd+" --resume 1\n")
