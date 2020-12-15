# -*- coding: utf-8 -*-
# %%
import os
import allel
import numpy as np
import pandas as pd
import time
import sys
import datetime

#parallel processing libraries
import multiprocessing as mp
from functools import partial # pool.map with multiple args
import subprocess as sp


#DL libraries
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

#test torch.float64 later (impact in accuracy and runtime)
torch.set_default_dtype(torch.float32)

############Learning options
n_masks = 500 #how many different masking ratios to try before checkpoint
#256 was the best batch size in TF1, test other batcsh size in pytorch
#batch_size=128
#batch_size=256
use_last_batch_for_validation=False

############data encoding options
encode_inputs_to_binomial=True #True: output layer will hav same format as output, False: use dosages directly as input (fewer nodes, less memory)
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
par_mask_proc=mp.cpu_count()
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


def convert_dosage_to_binomial(GT_input):
    GT = np.reshape(GT_input,[GT_input.shape[0],GT_input.shape[1]])
    ds_to_allel_presence_prob = {0:[1,0], 1:[1,1], 2:[0,1], -1: [0,0]}
    results = np.zeros([GT.shape[0],GT.shape[1],2])
    for old, new in ds_to_allel_presence_prob.items():
        results[GT == old] = new
    return results

def extract_genotypes_allel(my_path, extract_pos=False, unique=False):
    my_GT = []
    callset = allel.read_vcf(my_path, tabix=tabix_path,alt_number=2)
    my_GT = callset['calldata/GT']
    sample_ids = callset['samples']
    my_GT = allel.GenotypeArray(my_GT,dtype='i1')
    ac = my_GT.count_alleles()
    my_GT = my_GT.to_n_alt(fill=-1)
    
    print("my_GT SHAPE:",my_GT.shape)
    my_GT = my_GT.T
    
    #print("first 10 my_path:", my_GT[0,0:10])
    #print("last 10 my_path:", my_GT[0,-10:])
    
    results = np.reshape(my_GT, (my_GT.shape[0],my_GT.shape[1],1))        
    #print("first 10 my_path after reshape:", results[0,0:10])
    #print("last 10 my_path after reshape:", results[0,-10:])
        
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
    #print("first 10 my_path after reshape and unique:", results[0,0:10])
    #print("last 10 my_path after reshape and unique:", results[0,-10:])
    
    if(extract_pos==True):
        return sample_ids, results, MAF, callset['variants/POS']
    return sample_ids, results, MAF



def calculate_alpha(y_true,flip=False):
    
    ys = convert_dosage_to_binomial(y_true)
    ys = np.reshape(ys, [ys.shape[0], ys.shape[1]*ys.shape[2]])
    #w0=np.mean(1-ys, axis=0)
    w1=np.mean(ys, axis=0)
    #alpha = 1/0.0001+np.min(np.stack([w0,w1]), axis=0)
    alpha1 = np.expand_dims(w1, axis=0)
    if(flip==True):
        alpha1 = 1-alpha1
    #alpha = np.nan_to_num(alpha, nan=0.0001, posinf=0.0001, neginf=0.0001)
    alpha1 = Variable(torch.from_numpy(alpha1).float()).cuda()
    #for float64 precision
    #alpha1 = Variable(torch.from_numpy(alpha1).double()).cuda()
    return alpha1

def filter_by_MAF(ys, MAF):
    original_shape = ys.shape
    
    ys_values = ys.copy().transpose((1,0,2))
    
    indexes = list(range(len(MAF)))
    ys_dict = dict(zip(indexes,ys_values))
    
    MAF_a = np.array(MAF)
    filtered_mask = MAF_a >= min_MAF
    indexes = np.array(indexes)
    filtered_keys = indexes[filtered_mask]
    
    subdict = {x: ys_dict[x] for x in filtered_keys}
    sorted_subdict = dict(sorted(subdict.items(), key=lambda item: item[0]))
    mapped_ys = np.array(list(sorted_subdict.values())).transpose((1,0,2))
    
    print(mapped_ys.shape)
    print("UNIQUE keys", len(np.unique(filtered_keys)))
    mapped_ys = np.reshape(mapped_ys, [original_shape[0],len(filtered_keys), original_shape[2]])
   
    return mapped_ys


def mask_data_per_sample_parallel(mydata, mask_rate=0.9):

    if(verbose>0):
        print("Data to mask shape:", mydata.shape, flush=False)
        
    nmask = int(round(len(mydata[0])*mask_rate))
    my_mask=[0,0]
    if(encode_inputs_to_binomial==False):
        my_mask=[0]

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
def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 1) # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat)).to('cuda:0')
    return torch.mean(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

# define the sparse loss function
def sparse_loss(rho, data, model_children):
    #apply sparsity on encoder and decoder
    #values = data
    #loss = 0
    #for i in range(len(model_children)):
    #    values = model_children[i](values)
    #    loss += kl_divergence(rho, values)
    #aply sparsity on encoder only
    encoded_values = model_children[0](data)
    loss = kl_divergence(rho, encoded_values)
    return loss    

# define L1_loss
def l1_loss(model):
    l1_regularization = 0.
    l1_loss = nn.L1Loss()
    for name, param in model.named_parameters():
        if 'weight' in name:
            #faster
            l1_regularization = l1_regularization + l1_loss(param, target=torch.zeros_like(param))
            #torch.norm(x, ord) = sum(abs(x)**ord)**(1./ord)
            #l1_regularization = l1_regularization + torch.norm(param, 1)
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
        #input_dim – size of each input sample
        #output_dim – size of each output sample
        #n_layers - total number of layers
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
        return x    

def focal_loss(y_pred, y_true, gamma=3, alpha=None):
    loss=nn.BCELoss(reduction='none')
    cross_entropy_loss = loss(y_pred, y_true)
    p_t = ((y_true * y_pred) + ((1 - y_true) * (1 - y_pred)))

    modulating_factor = 1.0
    if gamma:
        modulating_factor = torch.pow(1.0 - p_t, gamma)

    alpha_weight_factor = 1.0
    if alpha is not None:
        alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))
    
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * cross_entropy_loss)
    
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

def main(ar):

    print("Input file:", ar.input)
    print("Minimum mask ratio:", ar.min_mask)
    print("Maximum mask ratio:", ar.max_mask)
    #input file in vcf format (e.g HRC ground truth VMV files)

    sample_ids, y_true, MAF, pos = extract_genotypes_allel(ar.input, extract_pos=True, unique=True)
    print("Input dims:", y_true.shape)
    
    if(min_MAF>0):
        filtered_y_true = filter_by_MAF(y_true, MAF)
        print("FILTERED OUTPUT LAYER WITH MAF THRESHOLD", min_MAF)
    else:
        filtered_y_true = y_true
        
    if(use_last_batch_for_validation==True):
        y_train = y_true[0:-batch_size]
        y_val = y_true[-batch_size:]
        filtered_y_train = filtered_y_true[0:-batch_size]
        filtered_y_val = filtered_y_true[-batch_size:]
    else:
        y_train = y_true
        filtered_y_train = filtered_y_true
        
    
    #mask ratios gradually increase mask rate starting from minimum mask value until it reaches the maximum values
    if(ar.min_mask!=ar.max_mask):
        mask_ratio_list = np.linspace(ar.min_mask, ar.max_mask, n_masks)
        mask_ratio_list[-1] = ar.max_mask
    else:
        #if fixed masking value is provided, just repeat instead of increase
        mask_ratio_list = np.repeat(ar.min_mask, n_masks)

    mask_ratio_list = np.round(mask_ratio_list,4)
    print("mask_ratios to run:", len(mask_ratio_list))        
    
    ni=y_train.shape[1]
    no=filtered_y_train.shape[1]*2
        
    if(encode_inputs_to_binomial==True):
        ni=ni*2
        
    n=len(y_true)
    
    #print("TRAIN POS:", pos)
    #print("VAL POS:", val_pos)    
    print("NI:",ni, "NO:", no)
    
    #best model hyperparameters from TF1 grid search
    learning_rate = ar.learn_rate
    BETA = ar.beta
    RHO = ar.rho
    L1 = ar.l1
    L2 = ar.l2
    GAMMA = ar.gamma
    loss_type = ar.loss_type
    disable_alpha = ar.disable_alpha
    #learning rate decay, initially not applied in TF1, default in pytorch is: decayRate = 0.96
    decayRate = ar.decay_rate
    #for testing different layer sizes
    size_ratio = ar.size_ratio
    #include activation function and optimizer as hyperparmater
    optimizer_type = ar.optimizer
    act = ar.activation
    n_layers = ar.n_layers
    batch_size = ar.batch_size
    start = 0 #if resuming, the number of the start epoch will change

    #will only reach max epochs if early stop won't reach a plateau
    max_epochs=ar.max_epochs

    #SPARSE AUTOENCODER
    autoencoder = Autoencoder(input_dim=ni, output_dim=no, n_layers=n_layers, size_ratio=size_ratio, activation=act).cuda()
    
    criterion = nn.BCELoss()
    optimizer = get_optimizer(autoencoder.parameters(), learning_rate, L2, optimizer_type=optimizer_type)
    
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    
    # get the layers as a list to apply sparsity penalty later
    model_children = list(autoencoder.children()) 
    print(model_children)
    if(ar.model_dir=='auto'):
        model_dir="./IMPUTATOR" + "_" + os.path.basename(ar.input)
    else:
        model_dir=arg.model_dir

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model_dir+'/'+ar.model_id+'.pth'
    print("Model save path:", model_path)

    hp_path = model_dir+'/'+ar.model_id+'_param.py'

    #TRAINING RESUME FEATURE ADDED: loads weights from previously trained model. Note that the current model must have the same architecture as the previous one.
    if(ar.resume==True and (not os.path.exists(model_path) or not os.path.exists(hp_path))):
        print("WARNING: model path doesn't exist:", model_path+" (and/or its respective *_param.py)", "\nYou set --resume=True but there is no model to resume from. Make sure you provided the correct path. The model will be trained from scratch.")
    if(ar.resume==True and os.path.exists(model_path) and os.path.exists(hp_path)):
        print("Resume mode activated (--resume) and found pre-existing model weights at", model_path, "\nLoading weights")
        autoencoder.load_state_dict(torch.load(model_path))
        import importlib.util #only needed if resuming
        spec = importlib.util.spec_from_file_location(ar.model_id+'_param', hp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        start=module.last_epoch

    with open(hp_path, 'w') as param_file:
        param_file.write("n_layers = "+str(n_layers)+"\n")
        param_file.write("size_ratio = "+str(size_ratio)+"\n")
        param_file.write("activation = \'"+act+"\'"+"\n")
    print("Inference parameters saved at:", hp_path)
    
    if(use_last_batch_for_validation==True):
        print("shape val (input):", y_val.shape)
        print("shape val (output):", filtered_y_val.shape)        
    print("shape train (input):", y_train.shape)
    print("shape train (output):", filtered_y_train.shape)
    
    total_batch = int((n/batch_size)-0.5)
    avg_loss = np.inf
    tmp_loss = 0
    
    if(encode_inputs_to_binomial==True):
        data_obs = convert_dosage_to_binomial(y_train.copy())
        filtered_y_train = convert_dosage_to_binomial(filtered_y_train)
        
        if(use_last_batch_for_validation==True):
            val_data_obs = convert_dosage_to_binomial(y_val.copy())
            filtered_y_val = convert_dosage_to_binomial(filtered_y_val)
    else:
        data_obs = y_train.copy()
        if(use_last_batch_for_validation==True):
            val_data_obs = y_val.copy()

    #for focal loss
    if(disable_alpha == True):
        alphas = None
    else:
        alphas = calculate_alpha(filtered_y_true)
    
    startTime = time.time()
    i=0
    
    for epoch in range(start,max_epochs):
        epochStart = time.time()
        epoch_loss=0
        
        #prepare data, do masking
        print("MASK RATIO:",mask_ratio_list[i])
        xs = mask_data_per_sample_parallel(data_obs.copy(), mask_rate=mask_ratio_list[i])
        xs = flatten_data(xs)
        ys = flatten_data(filtered_y_train.copy())
        randomize = np.random.rand(len(ys)).argsort()
        xs=xs[randomize]
        ys=ys[randomize]
        
        i+=1
        if(i==n_masks):
            i=0
        for batch_i in range(total_batch):
            
            #prepare data
            batch_features = [xs[batch_i*batch_size:(batch_i+1)*batch_size], 
                              ys[batch_i*batch_size:(batch_i+1)*batch_size]]
            
            #for float64
            #masked_data = Variable(torch.from_numpy(batch_features[0]).double()).cuda()
            #true_data = Variable(torch.from_numpy(batch_features[1]).double()).cuda()
            
            #for float32
            masked_data = Variable(torch.from_numpy(batch_features[0]).float()).cuda()
            true_data = Variable(torch.from_numpy(batch_features[1]).float()).cuda()
            
            #forward propagation
            reconstructed = autoencoder(masked_data)
           
            #focal loss
            if(loss_type=='FL'):
                loss = focal_loss(reconstructed, true_data, gamma=GAMMA, alpha=alphas)
            else:
                #CE (log loss)
                loss = criterion(reconstructed, true_data)
            
            #if applying L1 regularizaton
            if L1 > 0:
                l1_sparsity = l1_loss(autoencoder)
                loss = loss + l1_sparsity
            
            #if applying KL divergence regularization
            if BETA > 0:
                kl_sparsity = sparse_loss(RHO, true_data, model_children)
                loss = loss + kl_sparsity
                
            #backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss+=loss.data
        
        if(use_last_batch_for_validation==True):
            val_xs = mask_data_per_sample_parallel(val_data_obs.copy(), mask_rate=mask_ratio_list[i])
            val_xs = flatten_data(val_xs)
            val_ys = flatten_data(filtered_y_val.copy())
            val_masked_data = Variable(torch.from_numpy(val_xs).float()).cuda()
            val_true_data = Variable(torch.from_numpy(val_ys).float()).cuda()
            val_reconstructed = autoencoder(val_masked_data)
            val_loss = criterion(val_reconstructed, val_true_data)
            print('epoch [{}/{}], validation loss:{:.4f}'.format(epoch + 1, max_epochs,val_loss.data))
                

                
        tmp_loss += epoch_loss
        epochTime = (time.time()-epochStart)
                
        print('epoch [{}/{}], epoch time:{:.4f}, loss:{:.4f}'.format(epoch + 1, max_epochs,epochTime, epoch_loss), flush=True)
        
        if((epoch+1) % n_masks == 0):
            if(avg_loss > tmp_loss):
                print("Loss improved from", avg_loss, "to", tmp_loss)
                avg_loss = tmp_loss
                tmp_loss = 0
                print("Saving model to", model_path)
                torch.save(autoencoder.state_dict(), model_path)
                with open(hp_path, 'a') as param_file:
                    param_file.write("last_epoch = "+str(epoch+1)+"\n")
            else:
                print("Early stoping, no improvement observed. Previous loss:", avg_loss, "Currentloss:", tmp_loss)
                print("Best model at", model_path)
                with open(hp_path, 'a') as param_file:
                    param_file.write("early_stop = "+str(epoch+1)+"\n")
                break
            #exponentially decrease learning rate in each checkpoint
            if(decayRate>0):
                my_lr_scheduler.step()

    executionTime = (time.time() - startTime)

    print('Execution time in seconds: ' + str(executionTime))
    print('Run time per epoch: ' + str(executionTime/epoch))

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input", type=str, help="[str] Input file (ground truth) in VCF format")
    parser.add_argument("-N", "--min_mask", type=float, help="[float] Minimum masking ratio", default=0.8)
    parser.add_argument("-M", "--max_mask", type=float, help="[float] Maximum masking ratio", default=0.99)
    parser.add_argument("-L", "--l1", type=float, help="[float] L1 regularization scaling factor", default=1e-9)
    parser.add_argument("-W", "--l2", type=float, help="[float] L2 regularization scaling factor (a.k.a. weight decay)", default=1e-5)
    parser.add_argument("-B", "--beta", type=float, help="[float] Beta scaling factor for sparsity loss (KL divergence)", default=0.)
    parser.add_argument("-R", "--rho", type=float, help="[float] Rho desired mean activation for sparsity loss (KL divergence)", default=0.05)
    parser.add_argument("-G", "--gamma", type=float, help="[float] gamma modulating factor for focal loss", default=0.)
    parser.add_argument("-A", "--disable_alpha", type=int, help="[0 or 1]=[false or true] whether disable alpha scaling factor for focal loss", default=0)
    parser.add_argument("-C", "--learn_rate", type=float, help="[float] learning rate", default=0.001)
    parser.add_argument("-F", "--activation", type=str, help="[relu, leakyrelu, tanh, sigmoid] activation function type", default='relu')
    parser.add_argument("-O", "--optimizer", type=str, help="[adam, sgd, adadelta, adagrad] optimizer type", default='adam')
    parser.add_argument("-T", "--loss_type", type=str, help="[CE or FL] whether use CE for binary cross entropy or FL for focal loss", default='CE')
    parser.add_argument("-D", "--n_layers", type=int, help="[int, even number] total number of hidden layers", default=4)
    parser.add_argument("-S", "--size_ratio", type=float, help="[float(0-1]] size ratio for successive layer shrink (current layer size = previous layer size * size_ratio)", default=0.5)
    parser.add_argument("-E", "--decay_rate", type=float, help="[float[0-1]] learning rate decay ratio (0 = decay deabled)", default=0.)
    parser.add_argument("-H", "--model_id", type=str, help="[int/str] model id or name to use for saving the model", default='best_model')
    parser.add_argument("-J", "--model_dir", type=str, help="[str] path/directory to save the model", default='auto')
    parser.add_argument("-Z", "--batch_size", type=int, help="[int] batch size", default=256)
    parser.add_argument("-X", "--max_epochs", type=int, help="[int] maximum number of epochs if early stop criterion is not reached", default=20000)
    parser.add_argument("-U", "--resume", type=int, help="[0 or 1]=[false or true] whether enable resume mode: recover saved model (<model_id>.pth file) in the model folder and resume training from its saved weights.", default=0)

    args = parser.parse_args()
    print("ARGUMENTS", args)

    main(args)
