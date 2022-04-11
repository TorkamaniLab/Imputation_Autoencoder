# -*- coding: utf-8 -*-
# %%
import os
import allel
import numpy as np
import pandas as pd
import time
import sys
import datetime
import subprocess

#parallel processing libraries
import multiprocessing as mp
from functools import partial # pool.map with multiple args
import subprocess as sp
import random

#DL libraries
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

#optuna
import optuna
import logging
from joblib import parallel_backend
from multiprocessing import Manager
import plotly
from custom_samplers import SimulatedAnnealingSampler
import gc

#from inference function utility
from genotype_conversions_dip_oh import convert_gt_to_int

print("Pytorch version is:", torch.__version__)
print("Cuda version is:", torch.version.cuda)
print("cuDNN version is :", torch.backends.cudnn.version())
print("Arch version is :", torch._C._cuda_getArchFlags())

#test torch.float64 later (impact in accuracy and runtime)
torch.set_default_dtype(torch.float32)

############data encoding options
min_MAF=0.0

#Debugging options
verbose=0

#change this only for banchmarking multiprocessing speed
par_mask_method = "threadpool" #["serial","threadpool","thread","joblib","ray"] "serial" to disable, parallel masking method, threadpool has been the fastest I found
if(par_mask_method == "joblib"):
    import joblib
elif(par_mask_method == "thread"):
    import threading, queue
elif(par_mask_method == "ray"):
    import ray
do_numpy_masking=True
CPU_GPU_par=True

#set max 40 or less CPUs for masking, avoid interprocess and CPU-GPU communication overload
par_mask_proc=min(40,mp.cpu_count())
#

def cmd_exists(cmd):
    for path in os.environ["PATH"].split(os.pathsep):
        if(os.access(os.path.join(path, cmd), os.X_OK) == True):
            return path+"/"+cmd
    return False

tabix_path = cmd_exists('tabix')

if(tabix_path == False):
    #tabix_path='/opt/applications/samtools/1.9/gnu/bin/tabix'
    tabix_path='/home/rdias/bin/tabix'
    if(os.path.isfile(tabix_path)==False):
        print("WARNING!!! Tabix not found, VCF extraction may be slow, VCF loading may crash!!!")
    else:
        print ("Using tabix at:", tabix_path)
else:
    print ("Using tabix at:", tabix_path)

def convert_diplotypes_to_onehot(GT):
    label_to_allel_presence_prob = {'AA':[1,0,1,0], 'AB':[1,0,0,1],'BA':[0,1,1,0], 'BB':[0,1,0,1], 'missing':[0,0,0,0]}
    label_to_dip = {'AA':[0,0], 'AB':[0,1], 'BA':[1,0], 'BB':[1,1], 'missing':[-1,-1]}
    results = np.zeros([GT.shape[0], GT.shape[1], 4])
    
    for key, old in label_to_dip.items():
        boolean_mask = GT == old
        boolean_mask = boolean_mask[:,:,0] * boolean_mask[:,:,1]
        new = label_to_allel_presence_prob[key]
        results[boolean_mask] = new

    return results


def extract_genotypes_allel(my_path, extract_pos=False, unique=False):
    my_GT = []
    callset = allel.read_vcf(my_path, tabix=tabix_path,alt_number=1)
    my_GT = callset['calldata/GT']
    sample_ids = callset['samples']
    my_GT = allel.GenotypeArray(my_GT,dtype='i1')
    ac = my_GT.count_alleles()
    
    my_GT = my_GT.transpose((1,0,2)) #sampleXvariant#allele
    results = my_GT
    
    print("my_GT SHAPE:",my_GT.shape)
    
    #REF,ALT frequecnies
    MAF = ac.to_frequencies()
    #ALT only frequecnies
    a1 = np.round(MAF[:,0],6)
    a2 = np.round(MAF[:,1],6)
    MAF = np.min(np.stack([a1,a2]), axis=0)

    #np.savetxt('original_ytrue_train_sample1_before_unique.txt', results[0])
    #unique samples only
    if(unique==True):
        _, unique_indexes = np.unique(my_GT, axis=0, return_index=True)
        results = results[unique_indexes]
        sample_ids = sample_ids[unique_indexes]
    
    if(extract_pos==True):
        return sample_ids, results, MAF, callset['variants/POS']
    return sample_ids, results, MAF


def calculate_alpha(y_true,flip=False, gpu='cuda'):
    
    ys = convert_diplotypes_to_onehot(y_true)
    ys = np.reshape(ys, [ys.shape[0], ys.shape[1]*ys.shape[2]])
    #w0=np.mean(1-ys, axis=0)
    w1=np.mean(ys, axis=0)
    #alpha = 1/0.0001+np.min(np.stack([w0,w1]), axis=0)
    alpha1 = np.expand_dims(w1, axis=0)
    if(flip==True):
        alpha1 = 1-alpha1
    #alpha = np.nan_to_num(alpha, nan=0.0001, posinf=0.0001, neginf=0.0001)
    alpha1 = Variable(torch.from_numpy(alpha1).float()).cuda(gpu)
    #for float64 precision
    #alpha1 = Variable(torch.from_numpy(alpha1).double()).cuda()
    return alpha1

def filter_by_MAF(ys, MAF, min_MAF, max_MAF):
    original_shape = ys.shape

    ys_values = ys.copy().transpose((1,0,2))

    indexes = list(range(len(MAF)))
    ys_dict = dict(zip(indexes,ys_values))

    MAF_a = np.array(MAF)
    filtered_mask = (MAF_a > min_MAF)*(MAF_a <= max_MAF)
    indexes = np.array(indexes)
    filtered_keys = indexes[filtered_mask]

    subdict = {x: ys_dict[x] for x in filtered_keys}
    sorted_subdict = dict(sorted(subdict.items(), key=lambda item: item[0]))
    mapped_ys = np.array(list(sorted_subdict.values())).transpose((1,0,2))

    print(mapped_ys.shape)
    #print("UNIQUE keys", len(np.unique(filtered_keys)))
    mapped_ys = np.reshape(mapped_ys, [original_shape[0],len(filtered_keys), original_shape[2]])

    return filtered_keys, mapped_ys


def mask_data_per_sample_parallel(mydata, mask_rate=0.9):

    if(verbose>0):
        print("Data to mask shape:", mydata.shape, flush=False)
        
    nmask = int(round(len(mydata[0])*mask_rate))
    
    my_mask=[0,0,0,0]
        
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    if(do_numpy_masking==True):
        m=len(mydata[0])
        s=len(mydata)
        arr=np.arange(m)
        # random matrix of indexes
        if(par_mask_method=="joblib"):
            
            def mask_worker(data,nmask,replace=False):
                n=len(data)
                m=len(data[0])
                inds=np.random.rand(n,m).argsort(axis=1)[:,0:nmask]
                data[np.arange(len(data))[:, None], inds] = my_mask
                return data
            
            result=joblib.Parallel(n_jobs=par_mask_proc)(joblib.delayed(mask_worker)(i,nmask) for i in chunks(mydata,par_mask_proc))
            mydata=np.array(result)
                       
        elif(par_mask_method=="thread"):
         
            proc = []
            q = queue.Queue()
            
            def mask_worker(data,q):
                n=len(data)
                m=len(data[0])
                inds=np.random.rand(n,m).argsort(axis=1)[:,0:nmask]
                data[np.arange(len(data))[:, None], inds] = my_mask
                q.put(data)
            
            for i in chunks(mydata,par_mask_proc):
                p = threading.Thread(target=mask_worker, args=(i,q))
                p.Daemon = True
                proc.append(p)
                               
            for i in chunks(proc,par_mask_proc):
                for j in i:
                    j.start()
                for j in i:
                    j.join()
            
            mydata = [q.get() for i in proc]
            
            mydata = np.array(mydata)
          
        elif(par_mask_method=="threadpool" or par_mask_method=="pool"):
                       
            def mask_worker(data):
                n=len(data)
                m=len(data[0])
                inds=np.random.rand(n,m).argsort(axis=1)[:,0:nmask]
                data[np.arange(len(data))[:, None], inds] = my_mask
                return data
            
            pool = mp.pool.ThreadPool(par_mask_proc)
            result = pool.map(mask_worker,chunks(mydata,par_mask_proc))
            pool.close()
            pool.join()    
            result = [val for sublist in result for val in sublist]
            mydata = np.array(result)
            
        elif(par_mask_method=="ray"):
            #a little slower if you restart ray every time
            #ray.shutdown()
            #ray.init()
            @ray.remote
            def mask_worker(data):
                n=len(data)
                m=len(data[0])
                inds=np.random.rand(n,m).argsort(axis=1)[:,0:nmask]
                #fails if we try to overwrite, solution bellow
                data.setflags(write=1)                
                data[np.arange(len(data))[:, None], inds] = my_mask               

            futures = [mask_worker.remote(i) for i in chunks(mydata,par_mask_proc)]
            result = ray.get(futures)
            result = np.array(result)
            mydata = result
            #ray.shutdown()
            
        else:
            m=len(mydata[0])
            inds=np.random.rand(s,m).argsort(axis=1)[:,0:nmask]
            #slower
            #inds=np.stack([np.random.choice(np.arange(m),size=nmask,replace=False) for i in range(s)])
            mydata[np.arange(s)[:, None], inds] = my_mask
    else:
        #REAAAAAAALLY SLOOOOOOW, not using this anymore, keep just for debugging purposes
        j = 0
        while j < len(mydata):
            #redefines which variants will be masked for every new sample
            maskindex = random.sample(range(0, len(mydata[0]-1)), nmask) 
            for i in maskindex:
                mydata[j][i]=my_mask
            j=j+1

    if(verbose>0):
        print("Masked shape:", mydata.shape, flush=False)
    return mydata 


#SPARSE DENOISING AUTOENCODER, DIMENSIONALITY REDUCTION WITH SPARSITY LOSS (KL-DIVERGENCE)
def kl_divergence(rho, rho_hat, gpu=0):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 1) # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat)).cuda(gpu)
    return torch.mean(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

# define the sparse loss function
def sparse_loss(rho, data, model_children, gpu=0):
    #apply sparsity on encoder and decoder
    #values = data
    #loss = 0
    #for i in range(len(model_children)):
    #    values = model_children[i](values)
    #    loss += kl_divergence(rho, values)
    #apply sparsity on encoder only
    encoded_values = model_children[0](data)
    loss = kl_divergence(rho, encoded_values, gpu=gpu)
    return loss    

# define L1_loss
def l1_loss(model):
    l1_regularization = torch.tensor(0., requires_grad=True)
    #l1_loss = nn.L1Loss()
    for name, param in model.named_parameters():
        if 'weight' in name:
            #faster and use less VRAM than l1_loss
            l1_regularization = l1_regularization + param.abs().mean()
            #l1_regularization = l1_regularization + l1_loss(param, target=torch.zeros_like(param))
    return l1_regularization


#sparse autoencoder, keep full hidden layer size equal output_dim, then apply sparsity in the loss later    
class Autoencoder(nn.Module):
    def __init__(self,input_dim, output_dim, n_layers=4, size_ratio=0.5, activation='relu'):
        super(Autoencoder, self).__init__()

        def get_activation(activation):

            if(activation=='relu'):
                return nn.ReLU(True)
            elif(activation=='tanh'):
                return nn.Tanh()
            elif(activation=='sigmoid'):
                return nn.Sigmoid()
            elif(activation=='leakyrelu'):
                return torch.nn.LeakyReLU()
        
        #nn.Linear Parameters
        #input_dim = size of each input sample
        #output_dim = size of each output sample
        #n_layers = total number of layers
        #size_ratio=1.0 keep full layer size for encoding hidden layers
        
        encoder_layers = []
        
        in_size_list = [input_dim]
        out_size_list = [output_dim]
                
        for i in range(int(n_layers/2)):
            out_size_list += [int(out_size_list[i]*size_ratio)]
            encoder_layers += [nn.Linear(in_size_list[i], out_size_list[i+1])]
            encoder_layers += [get_activation(activation)]
            in_size_list += [out_size_list[i+1]]
            

        decoder_layers = []
        out_size_list.reverse()
        
        for i in range(int(n_layers/2)-1):
            decoder_layers += [nn.Linear(out_size_list[i], out_size_list[i+1])]
            decoder_layers += [get_activation(activation)]
            
        decoder_layers += [nn.Linear(out_size_list[-2], output_dim)]  
        decoder_layers += [get_activation('sigmoid')]
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #avoiding error (Assertion `input_val >= zero && input_val <= one` failed.)
        #only supported by pytorch 1.8 or newer
        #x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        #supported by older pytorches
        x = torch.where(torch.isnan(x), torch.tensor(0.), x)
        x = torch.where(torch.isinf(x), torch.tensor(0.), x)
        return x    

def focal_loss(y_pred, y_true, gamma=3, alpha=None, inverse=False):
    loss=nn.BCELoss(reduction='none')

    cross_entropy_loss = loss(y_pred, y_true)
    p_t = ((y_true * y_pred) + ((1 - y_true) * (1 - y_pred)))

    modulating_factor = 1.0
    if gamma:
        modulating_factor = torch.pow(1.0 - p_t, gamma)

    alpha_weight_factor = 1.0
    if alpha is not None:
        alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))
    if(inverse==True):
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * cross_entropy_loss)
    else:
        focal_cross_entropy_loss = (modulating_factor * 1/alpha_weight_factor * cross_entropy_loss)
    
    return focal_cross_entropy_loss.mean()
            
def flatten_data(x):
    x = np.reshape(x, (x.shape[0],-1))
    return x    

def get_optimizer(parameters, learning_rate, L2, optimizer_type='adam'):
    if optimizer_type == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=L2)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=L2)
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=learning_rate, weight_decay=L2)
    elif optimizer_type == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=learning_rate, weight_decay=L2)
    elif optimizer_type == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=learning_rate, weight_decay=L2)
    else:
        print("oprimizer not supported:", optimizer_type, "setting adam as default")
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=L2)

def get_gpu_memory_available(gpu_id):
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MiB.
    """
    #1MiB =  1048576 bytes
    MiB = 1048576
    
    result = subprocess.check_output(
        [
            'nvidia-smi' , '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [x for x in result.strip().split('\n')]
    vram_used = float(gpu_memory[gpu_id])
    #print("GPU id:", str(gpu_id), "GPU RAM used, including extra driver buffer from nvidia-smi:", str(vram_used))
    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / MiB
    vram_available = total_mem-vram_used
    return vram_available


def main(args):
    global ar
    ar = args
    
    print("Input file:", ar.input)
    print("Minimum mask ratio:", ar.min_mask)
    print("Maximum mask ratio:", ar.max_mask)
    #input file in vcf format (e.g HRC ground truth VMV files)

    sample_ids, y_true, MAF, pos = extract_genotypes_allel(ar.input, extract_pos=True, unique=True)
    print("Input dims:", y_true.shape)
    
    global filtered_y_true
    if(ar.min_MAF>0 or ar.max_MAF<0.5):
        filtered_indexes, filtered_y_true = filter_by_MAF(y_true, MAF, ar.min_MAF, ar.max_MAF)
        pos = pos[filtered_indexes]
        print("FILTERED OUTPUT LAYER WITH MAF THRESHOLD", ar.min_MAF, "-", ar.max_MAF)
    else:
        filtered_y_true = y_true
    
    global filtered_y_train
    y_train = y_true
    filtered_y_train = filtered_y_true
    
    #number of input neurons
    global ni
    #number of output neurons
    global no
    ni=y_train.shape[1]*4
    no=filtered_y_train.shape[1]*4
        
    global n    
    n=len(y_true)
    
    print("Number of input neurons (NI):",ni, "Number of output neurons (NO):", no)

    #validation part
    global val_y_true
    val_sample_ids, val_y_true, val_MAF, val_pos = extract_genotypes_allel(ar.val_true, extract_pos=True, unique=False)
    val_y_true = np.sum(val_y_true, axis=2)
    print("val_y_true.shape", val_y_true.shape)
    print(np.min(val_y_true), np.max(val_y_true))

    #inference part
    global new_df_list
    new_df_list = []
    known_indexes_list = []
    for val_dataset in ar.val_input:
        new_df, _ = process_data(pos, val_dataset[0])
        new_df = flatten_data(new_df.copy())
        new_df_list.append(new_df)
    #inference part end

    global val_intersec_indexes
    global input_intersec_indexes

    _, _, input_intersec_indexes = np.intersect1d(val_pos, pos, return_indices=True)
    _, _, val_intersec_indexes = np.intersect1d(pos, val_pos, return_indices=True)
    val_y_true = val_y_true[:,val_intersec_indexes]
    #validation part end

    print("shape train (input):", y_train.shape)
    print("shape train (output):", filtered_y_train.shape)


    global data_obs
    data_obs = convert_diplotypes_to_onehot(y_train.copy())
    filtered_y_train = convert_diplotypes_to_onehot(filtered_y_train)
    
    #sampling resulution, completely random values or use spased intervals (carse-grid from my original method)
    global sampling_res
    sampling_res = ar.sampling_res

    class Objective:
        def __init__(self, gpu_queue):
            # Shared queue to manage GPU IDs.
            self.gpu_queue = gpu_queue
            #if we want to choose a gpu only when all GPUs are available to choose, use this variable
            self.total_gpus = torch.cuda.device_count()
       
        def __call__(self, trial):
                   
            #1 check if a gpu is available
            #2 remove gpu with largest amount of VRAM from the queue
            #3 estimate how much VRAM is needed by model
            #4 if have memory needed, build model, otherwise wait
            #5 train the model for 3 epochs (to allow full allocation of VRAM by model)
            #6 put gpu back in the queue, updating the VRAM available (subtracting the memory used from total VRAM)
            #7 continue training until done 
                                
            #1MiB =  1048576 bytes
            MiB = 1048576
            precision = 32
            model_id = trial.number

            #avoid interprocess conflicting start
            time.sleep(random.uniform(0, 1))
            #1 check if a gpu is available
            while True:
                #replace len(self.gpu_queue)>0 by len(self.gpu_queue)==self.total_gpus if you only want to choose a gpu when all of them are available to choose, avoids having one unterutilized gpu just because it wasn't yet
                if(len(self.gpu_queue)>0 and len(self.gpu_queue)>=(self.total_gpus-1)):
                    
                    my_gpu = max(self.gpu_queue, key=self.gpu_queue.get)
                    #2 remove gpu with largest amount of VRAM from the queue
                    print("Model", model_id, "taking GPU", my_gpu)
                    del self.gpu_queue[my_gpu]
                    break
                else:
                    time.sleep(10)

            global sampling_res
            if(sampling_res=="random" or sampling_res=="Random"):
                learning_rate = trial.suggest_float("learn_rate", 1e-5, 0.1, log=True)
                BETA = trial.suggest_float("beta", 0, 5, log=False)
                RHO = trial.suggest_float("rho", 0, 1, log=False)
                L1 = trial.suggest_float("L1", 1e-16, 1, log=True)
                L2 = trial.suggest_float("L2", 1e-16, 1, log=True)
                decayRate = trial.suggest_float("decay_rate", 0., 1., log=False)
                size_ratio = trial.suggest_float("size_ratio", 0.25, 1, log=False)
            elif(sampling_res=="grid" or sampling_res=="Grid" or sampling_res=="coarse_grid"):
                learning_rate = trial.suggest_categorical("learn_rate", [1e-1,1e-2,1e-3,1e-4,1e-5])
                BETA = trial.suggest_categorical("beta", [0,1e-1,1e-2,1e-3,1e-4,1e-5,1,5])
                RHO = trial.suggest_categorical("rho", [0.005, 0.01, 0.05, 0.1, 0.25, 0.5])
                L1 = trial.suggest_categorical("L1", [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8, 0.])
                L2 = trial.suggest_categorical("L2", [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8, 0.])
                decayRate = trial.suggest_categorical("decay_rate", [0.0, 0.25, 0.5, 0.75, 0.95, 0.99])
                size_ratio = trial.suggest_categorical("size_ratio", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25])
            else:
                print("sampling resolution not identified", sampling_res, "please provide either coarse_grid or random for the --sampling_res argument. Exiting...")
                sys.exit()
            
            loss_type = trial.suggest_categorical("loss_type", ['CE', 'FL'])
            if(loss_type == 'FL'):
                disable_alpha = trial.suggest_categorical("disable_alpha", [0., 1.])
                inverse_alpha = 0
                flip_alpha = 0
                if(disable_alpha == 0):
                    flip_alpha = trial.suggest_categorical("flip_alpha", [0, 1])
                    inverse_alpha = trial.suggest_categorical("inverse_alpha", [0, 1])
                if(sampling_res=="random" or sampling_res=="Random"):
                    GAMMA = trial.suggest_float("gamma", 0, 7, log=False)
                else:
                    GAMMA = trial.suggest_categorical("gamma", [0,0.5,1,2,3,4,5,7])
            else:
                disable_alpha = True
            
            optimizer_type = trial.suggest_categorical("optimizer_type", ['adam', 'sgd', 'adadelta', 'adagrad'])
            act = trial.suggest_categorical("activation", ['relu', 'leakyrelu', 'tanh', 'sigmoid'])
            n_layers = trial.suggest_categorical("n_layers", [2,4,6,8,10,12])
            batch_size = trial.suggest_categorical("batch_size", [64,128,256,512])

            start = 0 #if resuming, the number of the start epoch will change
            avg_loss = np.inf #if resuming, starting avg_loss will be the previous checkpoint's loss value
            tmp_loss = 0
            total_batch = int((n/batch_size)-0.5)

            #100 epochs was too small to see any improvemnt, had to fix it to a minimum of 500 epochs
            #max_epochs = trial.suggest_categorical("max_epochs", [100,300,500])
            max_epochs = 300

            print("model_id: ", model_id, "GPU:", my_gpu, "PARAMS", trial.params)
            
            #checkpoint every n_masks epochs
            n_masks = 100

            #mask ratios gradually increase mask rate starting from minimum mask value until it reaches the maximum values
            if(ar.min_mask!=ar.max_mask):
                mask_ratio_list = np.linspace(ar.min_mask, ar.max_mask, n_masks)
                mask_ratio_list[-1] = ar.max_mask
            else:
                #if fixed masking value is provided, just repeat instead of increase
                mask_ratio_list = np.repeat(ar.min_mask, n_masks)

            mask_ratio_list = np.round(mask_ratio_list,4)
            print("mask_ratios to run:", len(mask_ratio_list))        

            #3 estimate how much VRAM is needed by model
            params = ni**2 * n_layers + ni*n_layers
            params_est = params  * precision / 8 / MiB
            input_output_layer = ni * batch_size * 2 * precision / 8 / MiB
            l1_variables = ni**2 * precision / 8 / MiB
            optimizer_variables = params * 4 * precision / 8 / MiB + l1_variables
            #*5 = true, masked, predcted, weights, bias, loss, L1 loss, sparsity loss
            variables = ni * batch_size * 5 * precision / 8 / MiB
            reconstructed_est = variables
            vram_estimate = params_est + input_output_layer + variables + reconstructed_est + l1_variables + optimizer_variables
            
            #if model is too big for this GPU total VRAM, then prune it (extra 5G VRAM, avoid out of memory error)
            if((vram_estimate+5000) > torch.cuda.get_device_properties(my_gpu).total_memory / MiB):
                print("Model", model_id, "returning GPU", my_gpu)
                self.gpu_queue[my_gpu] = get_gpu_memory_available(my_gpu)
                raise optuna.TrialPruned()

            #4 if have memory needed, build model, otherwise wait
            while True:
                    
                if((vram_estimate+5000) <= get_gpu_memory_available(my_gpu)):
                    try:
                        #SPARSE AUTOENCODER
                        autoencoder = Autoencoder(input_dim=ni, output_dim=no, n_layers=n_layers, size_ratio=size_ratio, activation=act).cuda(my_gpu)
                    except ValueError as e:
                        #if hyperparameter combination makes the model fail totally (e.g. too extreme hyperparameters generates invalid architecture values)
                        del e
                        del autoencoder
                        gc.collect()
                        torch.cuda.empty_cache()
                        print("Model", model_id, "returning GPU", my_gpu)
                        self.gpu_queue[my_gpu] = get_gpu_memory_available(my_gpu)
                        raise optuna.TrialPruned()
                    except ZeroDivisionError as e:
                        #if combination of small size_ratio and big n_layers generate hidden layers with zero neurons
                        del e
                        del autoencoder
                        gc.collect()
                        torch.cuda.empty_cache()
                        print("Model", model_id, "returning GPU", my_gpu)
                        self.gpu_queue[my_gpu] = get_gpu_memory_available(my_gpu)
                        raise optuna.TrialPruned()
                    else:
                        #if it worked, keep gpu
                        break
                else:
                    #wait
                    time.sleep(10) 
                    #put gpu back to queue with updated VRAM value, 
                    print("Model", model_id, "returning GPU", my_gpu)
                    self.gpu_queue[my_gpu] = get_gpu_memory_available(my_gpu)
                    #and check if it is really the most available gpu/VRAM
                    my_gpu = max(self.gpu_queue, key=self.gpu_queue.get)
                    #remove gpu with largest amount of VRAM from the queue
                    print("Model", model_id, "taking GPU", my_gpu)
                    del self.gpu_queue[my_gpu]


            criterion = nn.BCELoss()
            optimizer = get_optimizer(autoencoder.parameters(), learning_rate, L2, optimizer_type=optimizer_type)

            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
            current_lr =  my_lr_scheduler.get_last_lr()
            print("Learning rate:", current_lr[0])

            # get the layers as a list to apply sparsity penalty later
            model_children = list(autoencoder.children()) 
            print(model_children)
            if(ar.model_dir=='auto'):
                model_dir=os.path.join(ar.trial_dir, "IMPUTATOR" + "_" + os.path.basename(ar.input))
            else:
                model_dir=arg.model_dir

            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            model_path = model_dir+'/model_'+str(model_id)+'.pth'
            print("Model save path:", model_path)

            hp_path = model_dir+'/model_'+str(model_id)+'_param.py'

            write_mode = 'w'
            
            with open(hp_path, write_mode) as param_file:
                param_file.write("n_layers = "+str(n_layers)+"\n")
                param_file.write("size_ratio = "+str(size_ratio)+"\n")
                param_file.write("activation = \'"+act+"\'"+"\n")
                if write_mode == 'w':
                    param_file.write("early_stop = 0\n")
            print("New inference parameters saved at:", hp_path)

            #for focal loss
            if(disable_alpha == True):
                alphas = None
            else:
                alphas = calculate_alpha(filtered_y_true, flip=flip_alpha, gpu=my_gpu)

            startTime = time.time()
            i=0

            def mask_worker(my_data_obs, my_filtered_y_train, my_mask_rate):
                xs = mask_data_per_sample_parallel(my_data_obs.copy(), my_mask_rate)
                xs = flatten_data(xs)
                ys = flatten_data(my_filtered_y_train.copy())
                randomize = np.random.rand(len(ys)).argsort()
                xs=xs[randomize]
                ys=ys[randomize]
                return [xs, ys]
            
            if(CPU_GPU_par==True):
                xs, ys = mask_worker(data_obs, filtered_y_train, mask_ratio_list[i])
            
            for epoch in range(start,max_epochs):
                
                #5 train the model for 3 epochs (to allow full allocation of VRAM by model)
                if(epoch==3):
                    #6 put gpu back in the queue, updating the VRAM available (subtracting the memory used from total VRAM)
                    print("Model", model_id, "returning GPU", my_gpu)
                    self.gpu_queue[my_gpu] = get_gpu_memory_available(my_gpu)
                
                #7 continue training until done 
                
                epochStart = time.time()
                epoch_loss=0
                #prepare data, do masking
                print("MASK RATIO:",mask_ratio_list[i])
                if(CPU_GPU_par==False):
                    #xs = mask_data_per_sample_parallel(data_obs.copy(), mask_rate=mask_ratio_list[i])
                    #xs = flatten_data(xs)
                    #ys = flatten_data(filtered_y_train.copy())
                    #randomize = np.random.rand(len(ys)).argsort()
                    #xs=xs[randomize]
                    #ys=ys[randomize]
                    xs, ys = mask_worker(data_obs, filtered_y_train, mask_ratio_list[i])
                    cpuStop = time.time()
                    cpuTime = (cpuStop-epochStart)
                else:
                    #running thread on the background
                    pool = mp.pool.ThreadPool(processes=1)
                    async_result = pool.apply_async(mask_worker, (data_obs, filtered_y_train, mask_ratio_list[i]))

                comTime = 0        
                gpuTime = 0

                i+=1
                if(i==n_masks):
                    i=0
                for batch_i in range(total_batch):

                    comStart = time.time()
                    #prepare data
                    batch_features = [xs[batch_i*batch_size:(batch_i+1)*batch_size],
                                      ys[batch_i*batch_size:(batch_i+1)*batch_size]]

                    #for float64
                    #masked_data = Variable(torch.from_numpy(batch_features[0]).double()).cuda()
                    #true_data = Variable(torch.from_numpy(batch_features[1]).double()).cuda()

                    #for float32
                    masked_data = Variable(torch.from_numpy(batch_features[0]).float()).cuda(my_gpu)
                    true_data = Variable(torch.from_numpy(batch_features[1]).float()).cuda(my_gpu)

                    gpuStart = time.time()
                    comTime += (gpuStart-comStart)

                    #forward propagation
                    reconstructed = autoencoder(masked_data)

                    #focal loss
                    if(loss_type=='FL'):
                        loss = focal_loss(reconstructed, true_data, gamma=GAMMA, alpha=alphas, inverse=inverse_alpha)
                    else:
                        #CE (log loss)
                        loss = criterion(reconstructed, true_data)

                    #if applying L1 regularizaton
                    if L1 > 0:
                        l1_sparsity = L1 * l1_loss(autoencoder)
                        loss = loss + l1_sparsity

                    #if applying KL divergence regularization
                    if BETA > 0:
                        kl_sparsity = BETA * sparse_loss(RHO, true_data, model_children, gpu=my_gpu)
                        loss = loss + kl_sparsity

                    #backward propagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    gpuTime += (time.time()-gpuStart)
                    epoch_loss+=loss.data


                tmp_loss += epoch_loss/total_batch
                
                if(CPU_GPU_par==True):
                    #wait for result from background thread
                    cpuStart = time.time()
                    xs, ys = async_result.get()
                    pool.close()
                    pool.terminate()
                    #pool.join()
                    cpuStop = time.time()
                    cpuTime = (cpuStop-cpuStart)
                    
                epochTime = (time.time()-epochStart)

                print("Model:",model_id,"GPU:",my_gpu,'epoch [{}/{}], epoch time:{:.4f}, CPU-task time:{:.4f}, GPU-task time:{:.4f}, CPU-GPU-communication time:{:.4f}, loss:{:.4f}'.format(epoch + 1, max_epochs,epochTime, cpuTime, gpuTime, comTime, epoch_loss/total_batch), flush=True)
                print("GPU queue:", self.gpu_queue)
                
                if(torch.isnan(epoch_loss)):
                    del optimizer
                    del loss
                    del autoencoder
                    del model_children
                    gc.collect()
                    torch.cuda.empty_cache()
                    if(epoch < 3 or my_gpu in self.gpu_queue.keys()):
                        print("Model", model_id, "returning GPU", my_gpu)
                        self.gpu_queue[my_gpu] = get_gpu_memory_available(my_gpu)
                    return 0.0

                #checkpoint every n_masks cycles
                if((epoch+1) % n_masks == 0):
                    if(avg_loss > tmp_loss):
                        print("Loss improved from", avg_loss, "to", tmp_loss)
                        avg_loss = tmp_loss
                        tmp_loss = 0
                        print("Saving model to", model_path)
                        torch.save(autoencoder.state_dict(), model_path)
                        with open(hp_path, 'a') as param_file:
                            param_file.write("last_epoch = "+str(epoch+1)+"\n")
                            param_file.write("avg_loss = "+str(avg_loss.item())+"\n")
                    else:
                        print("Early stoping, no improvement observed. Previous loss:", avg_loss, "Currentloss:", tmp_loss)
                        print("Best model at", model_path)
                        with open(hp_path, 'a') as param_file:
                            param_file.write("early_stop = "+str(epoch+1)+"\n")
                        break
                    #exponentially decrease learning rate in each checkpoint
                    if(decayRate>0):
                        my_lr_scheduler.step()
                        current_lr =  my_lr_scheduler.get_last_lr()
                        print("Current learning rate:",current_lr[0])
                
                
                    #inference part
                    autoencoder = autoencoder.cpu()
                    autoencoder.load_state_dict(torch.load(model_path))
                    r2_list = []
                    global input_intersec_indexes
                    global val_y_true
                    global new_df_list
                    for val_x in new_df_list:
                        new_df_tensor = Variable(torch.from_numpy(val_x).float())
                        y_pred = autoencoder(new_df_tensor)
                        y_pred_dosages = y_pred_to_dosages(y_pred)
                        y_pred_dosages = y_pred_dosages[:,input_intersec_indexes]
                        #inference part end
                        #validation part
                        r2_list.append(np.mean(pearson_r2(y_pred_dosages, val_y_true)))
                        
                    autoencoder = autoencoder.cuda(my_gpu)

                    r2 = np.mean(r2_list)
                    print("Model:",model_id,"GPU:",my_gpu,"Epoch", str(epoch+1), "Mean R-squared per variant for each val input", r2_list, "Mean:", r2)
                    trial.report(r2, epoch+1)
                    if trial.should_prune() and ar.pruning == 1:
                        del optimizer
                        del loss
                        del autoencoder
                        del model_children
                        gc.collect()
                        torch.cuda.empty_cache()
                        if(epoch < 3 or my_gpu in self.gpu_queue.keys()):
                            print("Model", model_id, "returning GPU", my_gpu)
                            self.gpu_queue[my_gpu] = get_gpu_memory_available(my_gpu)                        
                        raise optuna.TrialPruned()
                    #validation part end

            executionTime = (time.time() - startTime)

            print('Training time in seconds: ' + str(executionTime))
            print('Run time per epoch: ' + str(executionTime/epoch))

            startTime = time.time()

            #inference part
            autoencoder = autoencoder.cpu()
            autoencoder.load_state_dict(torch.load(model_path))
            r2_list = []

            for val_x in new_df_list:
                new_df_tensor = Variable(torch.from_numpy(val_x).float())
                y_pred = autoencoder(new_df_tensor)
                y_pred_dosages = y_pred_to_dosages(y_pred)
                y_pred_dosages = y_pred_dosages[:,input_intersec_indexes]
                #inference part end
                #validation part
                r2_list.append(np.mean(pearson_r2(y_pred_dosages, val_y_true)))
            
            r2 = np.mean(r2_list)
            print("Mean R-squared per variant for each val input", r2_list, "Mean:", r2)
            #validation part end

            executionTime = (time.time() - startTime)
            print('Validation time in seconds: ' + str(executionTime))
            
            del optimizer
            del loss
            del autoencoder
            del model_children
            gc.collect()
            torch.cuda.empty_cache()
            if(my_gpu in self.gpu_queue.keys()):
                self.gpu_queue[my_gpu]=get_gpu_memory_available(my_gpu)
            
            return r2
    #objective function end

    startTime = time.time()
    
    #https://optuna.readthedocs.io/en/stable/reference/samplers.html
    if(ar.sampler=="TPE" or ar.sampler=="Bayesian" or ar.sampler=="bayesian"):
        #bayesian
        sampler = optuna.samplers.TPESampler(seed=123)
    elif(ar.sampler=="Random" or ar.sampler=="random"):
        #Random search
        sampler = optuna.samplers.RandomSampler(seed=123)
    elif(ar.sampler=="SA" or ar.sampler=="sa"):
        sampler = SimulatedAnnealingSampler(seed=123)
    elif(ar.sampler=="CMA" or ar.sampler=="cma" or ar.sampler=="CmaEs" or ar.sampler=="CMA-ES"):
        sampler = optuna.samplers.CmaEsSampler(seed=123)        
    else:
        print("You provided", ar.sampler, "as sampler argument but it is not supported.")
        print("Supported samplers are either TPE, Random, or Base sampler. Exiting...")
        sys.exit()
    
    pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(n_min_trials=10), patience=ar.patience)
    
    #storage_name = "sqlite:///{}.db".format(ar.study_name)
    if not os.path.exists(ar.trial_dir):
        os.mkdir(ar.trial_dir)
        
        
    db_path = os.path.join(ar.trial_dir, ar.study_name+'.db')
    #sqlite causes sync deadlock errors, use mysql for parallel runs (multiple jobs per vmv)
    storage_name = "sqlite:///{}".format(db_path)

    #mysql for multiple jobs per vmv, you must have installed and configured mysql for this to work
    if(ar.mysql != ''):
        user_name, host_name = ar.mysql.split('@')
        import mysql.connector
        mydb = mysql.connector.connect(host=host_name,user=user_name)
        mycursor = mydb.cursor()
        study_name = ar.study_name.replace('-','_')
        print("user:", user_name, "host:", host_name, "database:", study_name)
        if(ar.resume==0):
            mycursor.execute("DROP DATABASE IF EXISTS " + study_name)
        mycursor.execute("CREATE DATABASE IF NOT EXISTS " + study_name)
        storage_name = "mysql://{}/{}".format(ar.mysql, study_name)
        print("Storage name (MySQL):", storage_name)
    if 'PGSQL_URL' in os.environ:
        ar.study_name = os.environ['STUDY']
        storage_name = os.environ['PGSQL_URL']
        print(f"Using POSTGRESQL to connect to study: {ar.study_name}")
    
    study = optuna.create_study(direction="maximize", sampler=sampler, 
                                study_name=ar.study_name, storage=storage_name,
                                load_if_exists=ar.resume, pruner=pruner)
    
    # Add stream handler of stdout to show the messages to see Optuna works expectedly.
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
   
    total_gpus = torch.cuda.device_count()
    #1 MiB = 1048576 bytes
    MiB = 1048576

    gpu_queue = {}

    for gpu_id in range(total_gpus):
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / MiB
        gpu_queue[gpu_id] = total_mem
    
    total_jobs = ar.max_models_per_gpu*total_gpus
    #will be deprecated, alternative exampl: https://gist.github.com/toshihikoyanase/593246a6d00d6e9ec52de7b7c0f346f3
    with Manager() as manager:
        with parallel_backend("multiprocessing", n_jobs=total_jobs):
            study.optimize(Objective(gpu_queue), n_trials=ar.n_trials, n_jobs=total_jobs, gc_after_trial=True, 
                           callbacks=[lambda study, trial: torch.cuda.empty_cache()])

    # Show results.
    my_param_name = "r2"
    result_path = os.path.join(ar.trial_dir, ar.study_name)
    result = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(result)
    
    result.to_csv(result_path+"_summary.csv")
    
    print("Best r-squared:", study.best_trial.value)
    
    executionTime = (time.time() - startTime)    
    print('Complete hyperparamater optimization time in seconds: ' + str(executionTime))
    
    if(ar.run_plots==1):
        print("Plotting...")
        startTime = time.time()

        fig = optuna.visualization.plot_optimization_history(study, target_name=my_param_name)
        fig.write_image(result_path+"_plot_optimization_history.pdf", format="pdf")

        #fig = optuna.visualization.plot_parallel_coordinate(study, target_name=my_param_name)
        #fig.write_image(result_path+"_plot_parallel_coordinate.pdf", format="pdf")

        fig = optuna.visualization.plot_param_importances(study, target_name=my_param_name)
        fig.write_image(result_path+"_plot_param_importances_r2.pdf", format="pdf")

        fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="run time")
        fig.write_image(result_path+"_plot_param_importances_runtime.pdf", format="pdf")

        fig = optuna.visualization.plot_edf(study, target_name=my_param_name)
        fig.write_image(result_path+"_plot_edf.pdf", format="pdf")

        fig = optuna.visualization.plot_intermediate_values(study)
        fig.write_image(result_path+"_plot_intermediate_values.pdf", format="pdf")    

        executionTime = (time.time() - startTime)    
        print('Plot time in seconds: ' + str(executionTime))
    

#modified from inference_function, we must update this if inf. function changes
def process_data(refpos, infile):

    refpos = pd.Series(refpos, index=range(len(refpos)))
    # 0      22065657
    # 1      22065697
    # 2      22065904

    #infile is the input file: genotype data set to be imputed
    df = pd.read_csv(infile, sep='\t', comment='#', header=None)

    inpos = pd.Series(range(len(df[1])), index=df[1])
    #0      22065657
    #1      22066211
    #2      22066363
    #genetic variants are rows and samples are columns

    new_df = np.zeros((df.shape[1]-9, len(refpos), 4))  # subjects, variants

    _, _, known_indexes = np.intersect1d(inpos.index.values, refpos.values, return_indices=True)
    known_indexes = 4 * known_indexes

    i = 9  # RR column index
    idx = 0
    inpos_keys = inpos.keys()
    while i < df.shape[1]:
        for refpos_idx, refpos_val in refpos.iteritems():
            if refpos_val in inpos_keys:
                myidx = inpos[refpos_val]
                #update convert_gt_to_int if input encoding changes
                new_df[idx][refpos_idx] = convert_gt_to_int(df[i][myidx][0:3], one_hot=True)
        i += 1
        idx += 1
    #the data needs to be flattened because the matrix multiplication step (x*W)
    return new_df, known_indexes    
    
def pearson_r2(x, y):
    #per variant: axis=0
    x_sum = np.sum(x, axis=0)
    y_sum = np.sum(y, axis=0)
    xy_sum = np.sum(np.multiply(x,y), axis=0)
    x_squared_sum = np.sum(np.power(x,2), axis=0)
    y_squared_sum = np.sum(np.power(y,2), axis=0)
    N=len(y)

    num=np.subtract(np.multiply(xy_sum, N), np.multiply(x_sum, y_sum) )
    den=np.multiply(x_squared_sum, N)
    den=np.subtract(den, np.power(x_sum,2))
    den2=np.multiply(y_squared_sum, N)
    den2=np.subtract(den2, np.power(y_sum,2))
    den=np.sqrt(np.multiply(den, den2))

    #if we want to calculate maf or filter by maf later
    #maf=y_sum/(len(y)*2)
    
    r2_per_variant=np.divide(num,den)
    #remove monomorphic variants from y_true
    r2_per_variant=r2_per_variant[y_sum > 0.0]

    r2_per_variant=np.nan_to_num(r2_per_variant, nan=0.0, posinf=0.0, neginf=0.0)
    r2_per_variant=np.power(r2_per_variant,2)
    r2_per_variant=np.round(r2_per_variant, decimals=6)
    
    return r2_per_variant
        
def y_pred_to_dosages(in_arr):

    #reshape to Sample by Variant by Allele (S x V x 2)
    in_arr = torch.reshape(in_arr, [-1,int(len(in_arr[0])/2),2])
    #convert allele dimension outputs into probabilitites
    in_arr = F.softmax(in_arr, dim=-1)
    in_arr = in_arr.cpu().detach().numpy()

    #sum probabilities of alternative alleles to get dosage
    Df1 = in_arr[:,0::2,1]+in_arr[:,1::2,1]
    Df1 = flatten_data(Df1)

    return Df1

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input", type=str, help="[str] Input file (ground truth) in VCF format")
    parser.add_argument("--val_input", type=str, help="[str] Validation input file (genotype array to be imputed) in VCF format, multiple values are accepted", action='append',nargs=1)
    parser.add_argument("--val_true", type=str, help="[str] Validation file (ground truth) in VCF format")
    parser.add_argument("-N", "--min_mask", type=float, help="[float] Minimum masking ratio", default=0.8)
    parser.add_argument("-M", "--max_mask", type=float, help="[float] Maximum masking ratio", default=0.99)
    parser.add_argument("--study_name", type=str, help="[str] trial or study name to use for saving the results", default="optuna-study")
    parser.add_argument("-J", "--model_dir", type=str, help="[str] path/directory to save the model", default='auto')
    parser.add_argument("--trial_dir", type=str, help="[str] path/directory to save the trial results, not compatible with --mysql", default='.')
    parser.add_argument("--min_MAF", type=float, help="[float] minimum minor allele frequency (MAF) to be in the output layer", default=0)
    parser.add_argument("--max_MAF", type=float, help="[float] maximum minor allele frequency (MAF) to be in the output layer", default=0.5)    
    parser.add_argument("-U", "--resume", type=int, help="[0 or 1]=[false or true] whether enable resume mode: recover saved model (<model_id>.pth file) in the model folder and resume training from its saved weights.", default=0)
    parser.add_argument("--n_trials", type=int, help="[int] How many hyperpearameter optimization trials to run.", default=5)
    parser.add_argument("--sampler", type=str, help="[str][TPE, Random, SA, CMA] Sampler algorithm. Supported values are 'TPE', 'Random', 'CMA', 'SA'", default='TPE')
    parser.add_argument("--patience", type=int, help="[int] patience for pruner.", default=0)
    parser.add_argument("--sampling_res", type=str, help="[str][coarse_grid, random] Sampling resolution", default='coarse_grid')
    parser.add_argument("--pruning", type=int, help="[0 or 1]=[false or true] whether enable model pruning, discarding models that are performing badly before they finish training. Criteria based on mediam performance of previous models.", default=1)
    parser.add_argument("--max_models_per_gpu", type=int, help="[int] Maximum number of models per GPU (recommended 6 to not overload the interprocess communication overhang and CPU). Number of GPUs is detected automatically.", default=6)
    parser.add_argument("--mysql", type=str, help="[str][<user>@<host>] If using MySQL, provide credentials (e.g. user@host or john@localhost)", default='')
    parser.add_argument("--run_plots", type=int, help="[0 or 1][true or false] Whether to generate plots by the end of the run", default=1)

    args = parser.parse_args()
    print("ARGUMENTS", args)

    main(args)
