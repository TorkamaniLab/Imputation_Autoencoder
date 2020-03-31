# coding: utf-8

import math #sqrt
import tensorflow as tf
import numpy as np #pip install numpy==1.16.4
import pandas as pd

import random #masking
# sorting results
from collections import defaultdict
from operator import itemgetter

import timeit #measure runtime

from scipy.stats import linregress

#arguments
import sys
import argparse

import operator #remove entire columns from 2d arrays

import subprocess as sp #run bash commands that are much faster than in python (i.e cut, grep, awk, etc)

import os

#parallel processing libraries
import multiprocessing
from functools import partial # pool.map with multiple args

###################################DEV_OPTIONS#############################################

############Performance options: data loading
do_parallel = False #load data and preprocess it in parallel, not recommended, scikit is faster
use_scikit_allel = True #use scikit-ellel to load the data, recommended, requires installation of skikit-allel package

############Performance options: masking
do_numpy_masking = True #array based method, much faster then nested for loops!
par_mask_proc = 20 #how many masking parallel threads or processes, only used if par_mask_method is not "serial"
par_mask_method = "threadpool" #["serial","threadpool","thread","joblib","ray"] "serial" to disable, parallel masking method, threadpool has been the fastest I found
do_parallel_numpy_per_cycle = True #whole cycle in parallel
do_parallel_gd_and_mask = False #instead of parallelizing masking only, run masking and gradient descent in parallel

############Performance options: tensorflow options
tf_precision = tf.float32 #either tf.float32 or tf.float64

############Performance options: future improvements
use_cuDF = False #TODO enable data loading directly in the GPU

############backup options and reporting options
save_model = True #[True,False] save final model generated after all training epochs, generating a checkpoint that can be loaded and continued later
save_activations = False #export activation values of the last hidden layer
save_pred = False #[True,False] save predicted results for each training epoch and for k-fold CV
resuming_step = 0 #number of first step if recovery mode True
save_summaries = False #save summaries for plotting in tensorboard
detailed_metrics = False #time consuming, calculate additional accuracy metrics for every variant and MAF threshold
report_perf_by_rarity = False #report performance for common versus rare variants
full_train_report = False #True: use full training set to report loss, accuracy, etc (a lot more VRAM needed) in the final result list, False: calculate average statistics per batch (less VRAM needed)

#lower and upper thresholds for rare VS common variants analyses
rare_threshold1 = 0
rare_threshold2 = 0.01
common_threshold1 = 0.01
common_threshold2 = 1

############Learning options
split_size = 100 #number of batches
training_epochs = 35000 #learning epochs (if fixed masking = True) or learning permutations (if fixed_masking = False), default 35000, number of epochs or data augmentation permutations (in data augmentation mode when fixed_masking = False)
#minimum training_epochs recommended for grid search training_epochs=499 (starts at 0)
last_batch_validation = False #when k==1, you may use the last batch for valitation if you want
all_sparse=True #set all hidden layers as sparse
average_loss=True #True/False use everage loss, otherwise total sum will be calculated
disable_alpha=False #disable alpha for debugging only
inverse_alpha=True #1/alpha
early_stop_begin=1 #after what epoch to start monitoring the early stop criteria
window=500 #stop criteria, threshold on how many epochs without improvement in average loss, if no improvent is observed, then interrupt training
hysteresis=0.001 #stop criteria, improvement ratio, extra room in the threshold of loss value to detect improvement, used to identify the beggining of a performance plateau

############Masking options
fixed_masking = False #True: mask variants only at the beggining of the training cycle, False: mask again with a different pattern after each iteration (data augmentation mode)
mask_per_sample = True #True: randomly mask genotypes per sample instead of mask the entire variant for all samples, False: mask the entire variant for all samples 
shuffle = True #Whether shuffle data or not at the begining of training cycle. Not necessary for online data augmentation.
repeat_cycles = 4 #how many times to repeat the masking rate
validate_after_epoch=False #after each epoch, apply a new masking pattern, then calculate validation accuracy on the new unseen mask pattern
validate_after_cycle=False
calculate_r2_per_epoch=False
calculate_acc_per_epoch=False

############debugging options
verbose=0

###################################OPTIONS#############################################

def cmd_exists(cmd):
    for path in os.environ["PATH"].split(os.pathsep):
        if(os.access(os.path.join(path, cmd), os.X_OK) == True):
            return path+"/"+cmd
    return False

if(use_scikit_allel==True):
    import allel
    tabix_path = cmd_exists('tabix')
    if(tabix_path == False):
        tabix_path='/opt/applications/samtools/1.9/gnu/bin/tabix'
        if(os.path.isfile(tabix_path)==False):
            print("WARNING!!! Tabix not found, VCF extraction may be slow, VCF loading may crash!!!")
        else:
            print ("Using tabix at:", tabix_path)
    else:
        print ("Using tabix at:", tabix_path)

if(use_scikit_allel == False):
    plink_path = cmd_exists('plink')
    if(plink_path == False):
        print("WARNING!!! Plink not found, MAF calculation may crash!!!")
    else:
        print ("Using plink at:", plink_path)

#global variables
MAF_all_var = [] #MAF calculated only once for all variants,remove redundancies
freq_all_var = [] #Frequency of the least frequent class for the encoded data
rare_indexes = [] #indexes of rare variants
common_indexes = [] #indexes of common variants
ncores = multiprocessing.cpu_count() #for parallel processing
config = tf.ConfigProto(log_device_placement=False)
config.intra_op_parallelism_threads = 0 #0=auto
config.inter_op_parallelism_threads = 0 #0=auto
config.gpu_options.allow_growth=True
model_index=0

#utility numbers used many times by many formulas
eps = tf.cast(1e-6, tf_precision)
one = tf.cast(1.0, tf_precision)
n1 = tf.cast(-1.0, tf_precision)


if(par_mask_method == "joblib"):
    import joblib
elif(par_mask_method == "thread"):
    import threading, queue
elif(par_mask_method == "ray"):
    import ray

def convert_gt_to_int(gt):

    genotype_to_int={'0/0': [1,0], '0|0': [1,0], '0/1': [1,1], '0|1':[1,1], '1/0':[1,1], '1|0':[1,1], '1/1':[0,1], '1|1':[0,1], './0':[0,0], './1':[0,0], './.':[0,0], '0/.':[0,0], '1/.':[0,0]}
    result=genotype_to_int[gt[0:3]]

    return result


def convert_genotypes_to_int(indexes, file):
    if(verbose>0):
        print("process:", multiprocessing.current_process().name, "arguments:", indexes, ":", file)
    
    j=0

    command = "cut -f " + str(indexes[0]) + "-" + str(indexes[len(indexes)-1]) + " " + file

    result = sp.check_output(command, encoding='UTF-8', shell=True)

    df = []
    first=True
    i=0
    for ln in result.split('\n'):
        i+=1
        if(not ln.startswith("#")):
            if(first==False and ln):
                tmp = ln.split('\t')
                #print(i, ": ", tmp, ": ", ln)
                df.append(tmp)
            else:
                first=False

    df = list(map(list, zip(*df)))   
    
    new_df = 0
    new_df = np.zeros((len(df),len(df[0]),2))

    i = 0 #RR column index
    j = 0 #RR row index
    idx = 0
    my_hom = 2
    
    if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
        my_hom = 1
    
    while i < len(df): #sample index, total-9 (9 for first comment columns)
        
        if(j==(len(df)-1)):
            print(j)
        
        j = 0 #variant index, while variant index less than total variant count
        while j < len(df[0]): #"|" is present when phased data is proved, "/" is usually unphased
            #print j

            df[i][j] = str(df[i][j])
            new_df[idx][j]=convert_gt_to_int(df[i][j][0:3])
            j += 1
        i += 1
        idx += 1

    return new_df.tolist()


#split input data into chunks so we can prepare batches in parallel
#new chunk, works on ND arrays instead of taking indexes
def chunk(data, ncores=multiprocessing.cpu_count()):

    n_max=10000
    n_min=100
    
    n=int(round(len(data)/ncores-0.5))
    if(n>n_max):
        n=n_max
    if(n<n_min):
        n=n_min
    chunks=[data[i:i + n] for i in range(0, len(data), n)]

    return chunks

def run_gd_step(my_sess, my_feed_dict, my_args):

    my_result = my_sess.run(my_args, feed_dict=my_feed_dict )
             
    return my_result
        

def run_epoch_worker(my_sess, X, Y, current_train_x, train_y, my_args, batch_size):

    if(verbose>0):
        print("current_train_x shape:", current_train_x.shape, flush=True)  
    my_result = []
    #for all batches, use current train_x current_train_y
    current_train_y = flatten_data_np(train_y)
    current_train_x = flatten_data_np(current_train_x)
    total_batch = int(current_train_x.shape[0] / batch_size)
    if(verbose>0):
        print("current_train_x shape after flatten:", current_train_x.shape, flush=True)    
    #process = multiprocessing.current_process()
    #my_id = process.pid
    for j in range(total_batch):
        if(verbose>0):
            print("Parallel gd, running batch: ", j,"batch size: ", batch_size,"nbatches",total_batch,flush=True)
        batch_x = current_train_x[j*batch_size:(j+1)*batch_size]
        batch_y = current_train_y[j*batch_size:(j+1)*batch_size]
        if(verbose>0):
            print("gd step starting", flush=True)
        my_tmp = my_sess.run(my_args, feed_dict={X: batch_x, Y: batch_y} )
        #my_tmp = run_gd_step(my_sess, my_feed_dict, my_args)

        if(verbose>0):
            print("Parallel gd, batch done: ", j, "result size", len(my_tmp),flush=True)

        my_result.append(my_tmp)
    #m_list[my_id] = my_result

    return my_result

def submit_mask_tasks(my_data, mask_rate):
    
    par_tasks = 1
    if(do_parallel_numpy_per_cycle==True):
        par_tasks += repeat_cycles
    
    pool = multiprocessing.pool.ThreadPool(par_tasks)
    #pool = multiprocessing.Pool(par_tasks)

    result = []
    for i in range(par_tasks):
        result.append(pool.apply_async(partial(mask_data_per_sample_parallel, mydata=np.copy(my_data), mask_rate=mask_rate),[i]))
    
    if(verbose>0):
        print("Runnning mask process, submitted.",flush=True)
    
    return result, pool

def retrieve_cycle_results(m_result, pool):

    results = []
    pool.close()
    pool.join()
    for thread in m_result:
        results.append(thread.get())

    return results

def run_parallel_gd_and_mask(sess, X, Y, current_train_x, my_data, mask_rate, my_args, batch_size):

    m_result, pool = submit_mask_tasks(my_data, mask_rate)
    gd_result = run_epoch_worker(sess, X,Y,current_train_x, my_data, my_args, batch_size)
    results = retrieve_cycle_results(m_result,pool)
    if(verbose>0):
        print("gd and mask in parallel done. Result shape:", len(gd_result), len(gd_result[0]),flush=True)

    return results, gd_result

#parse initial_masking_rate if fraction is provided
def convert_to_float(frac_str):
    if(',' in frac_str):
        a = frac_str.split(',')
        try:
            return [float(i) for i in a]
        except:
            print("Unable to convert keep probabilities into float:", frac_str)
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)        
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))


def process_data(file):

    #Header and column names start with hashtag, skip those
    ncols = pd.read_csv(file, sep='\t', comment='#',header=None, nrows=1)    
    ncols = ncols.shape[1]
    
    print("Processing input data.")
    print("use_scikit_allel: ", use_scikit_allel)

    n_samples = ncols-9
    #RR subtract 9 from the number of columns to get the number of SNPs, 
    #RR because the first 9 columns are variant information, not genotypes
    print("number of samples: ", n_samples)
    
    indexes = list(range(10,ncols+1)) #bash cut index is 1-based
    
    start = timeit.default_timer()

    #STRING_FROM_VCF -> DOSAGE -> PRESENCE_ENCODING
    #'0/0'           -> 0      -> [1,0]  (ref is present, alt is absent)
    #'1/0' or '0/1'  -> 1      -> [1,1]  (ref is present, alt is present)
    #'1/1'           -> 2      -> [0,1]  (ref is absent, alt is present)
    global MAF_all_var

    if(use_scikit_allel==True):
        callset = allel.read_vcf(file, tabix=tabix_path)
        my_GT = callset['calldata/GT']
        my_GT = allel.GenotypeArray(my_GT,dtype='i1')
        ac = my_GT.count_alleles()
        my_GT = my_GT.to_n_alt(fill=-1)
        my_GT = my_GT.T
        ds_to_allel_presence_prob = {0:[1,0], 1:[1,1], 2:[0,1], -1: [0,0]}
        results = np.zeros([my_GT.shape[0],my_GT.shape[1],2])
        for old, new in ds_to_allel_presence_prob.items():
            results[my_GT == old] = new

        #REF,ALT frequecnies
        MAF_all_var = ac.to_frequencies()
        #ALT only frequecnies
        MAF_all_var = np.round(MAF_all_var[:,1],6)

    elif(do_parallel==False):
        results = convert_genotypes_to_int(indexes, file)
        print( len(results), len(results[0]), len(results[0][0]))

    else:
        chunks = chunk(indexes, ncores )        

        pool = multiprocessing.Pool(ncores)

        results = pool.map(partial(convert_genotypes_to_int, file=file),chunks)
      
        pool.close()
        pool.join()

        print(len(results), len(results[0]), len(results[0][0]) , len(results[0][0][0]))
    
        results = [item for sublist in results for item in sublist]

    print(len(results), len(results[0]), len(results[0][0]) )

    print("This file contains {} features (SNPs) and {} samples (subjects)".format(len(results[0]), n_samples))
    
    indexes = list(range(len(results[0])))

    results = np.asarray(results)
    
    stop = timeit.default_timer()
    print('Time to load the data (sec): ', stop - start)
    
    start_time = timeit.default_timer()

    #calculates frequencies of least frequence class for using as alpha later
    global freq_all_var
    y_true = np.copy(results)
    y_true = flatten_data_np(y_true)
    freq1 = np.mean(y_true, 0) #frequency of ones along columns (axis 0, per variable)
    freq0 = np.mean(np.subtract(1.0, y_true),0) #frequency of zeros along columns (axis 0, per variable)
    freq01 = np.stack([freq0, freq1], 0) #stack M frequencies generated, resulting into a 2xM array
    freq_all_var = np.amin(freq01, 0) #return smallest value per colum
    #avoid div by zero
    freq_all_var = np.clip(freq_all_var,0.01,1)

    if(inverse_alpha == True):
        freq_all_var = np.divide(1.0,freq_all_var)

    freq_all_var = np.round(freq_all_var,6)

    if(use_scikit_allel == False):

        MAF_all_var = calculate_ref_MAF(file)

    #create list of variants that belong to output layer, 1 index per variant
    keep_indexes=list(range(left_buffer,len(MAF_all_var)-right_buffer))
    
    #find out the layer structure (either 2 or 3 features per variant)
    n=2

    #set start and end index of output layer
    start = 0
    if(left_buffer>0):
        start=left_buffer*n
    end=(len(MAF_all_var)-right_buffer)*n

    #create list of indexes of variants in output layer, 2 indexes per variant
    keep_indexes_vector=list(range(start,end))
            
    global rare_indexes
    global common_indexes

    #generate list of indexes of variants that are rare and common
    rare_indexes = filter_by_MAF_global(results, MAF_all_var, threshold1=rare_threshold1, threshold2=rare_threshold2)    
    common_indexes = filter_by_MAF_global(results, MAF_all_var, threshold1=common_threshold1, threshold2=common_threshold2)
    
    #map output layer indexes to the indexes of common/rare variants, generating intersection
    rare_intersect = [val for val in keep_indexes_vector if val in rare_indexes]
    common_intersect = [val for val in keep_indexes_vector if val in common_indexes]

    #save intersection replacing old list, subtracting left buffer
    rare_indexes = [x - start for x in rare_intersect]
    common_indexes = [x - start for x in common_intersect]
   
    #keep MAF info only for output layer variants
    MAF_all_var_tmp = [] 
    for i in keep_indexes:
        MAF_all_var_tmp.append(MAF_all_var[i])

    MAF_all_var = MAF_all_var_tmp

    print("ALLELE FREQUENCIES", MAF_all_var)
    print("LENGTH1", len(MAF_all_var)) 
    stop_time = timeit.default_timer()
    print('Time to calculate MAF (sec): ', stop_time - start_time)

    return results

def filter_by_MAF_global(x, MAFs, threshold1=0, threshold2=1):
    
    #don't do any filtering if the thresholds are 0 and 1
    if(threshold1==0 and threshold2==1):
        return x

    indexes_to_keep = []
    i = 0
    k = 0
    
    while i < len(MAFs):
        if(MAFs[i]>threshold1 and MAFs[i]<=threshold2):
            indexes_to_keep.append(k)
            indexes_to_keep.append(k+1)
        i += 1
        k += 2
        
    return indexes_to_keep


def read_MAF_file(file):

#   CHR         SNP   A1   A2          MAF  NCHROBS
#   9   rs1333039    G    C       0.3821    54330
#   9 rs138885769    T    C    0.0008099    54330
#   9 rs548022918    T    G     0.000589    54330
    pd.set_option('display.float_format', '{:.6f}'.format)

    maftable = pd.read_csv(file, sep='\s+', comment='#')
    maftable['MAF'] = maftable['MAF'].astype(float)

    result = maftable['MAF'].values.tolist()
    if(verbose>0):
        print("#####REF MAF#####",result)
    return result

def calculate_ref_MAF(refname):
    #plink --vcf HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf.clean4 --freq 
    outname=os.path.basename(refname)
    result = sp.check_output("plink --vcf "+refname+" --freq --out "+outname, encoding='UTF-8', shell=True)

    MAF_all_var = read_MAF_file(outname+".frq")

    return MAF_all_var

def run_custom_cmd():
    #bash *VMV.2_val.sh
    result = sp.check_output(custom_cmd, encoding='UTF-8', shell=True)

    return result

def mask_data_per_sample_parallel(i,mydata, mask_rate=0.9):

    if(verbose>0):
        print("Data to mask shape:", mydata.shape, flush=False)
        
    nmask = int(round(len(mydata[0])*mask_rate))
    my_mask=[0,0]

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
                if(mask_per_sample==False):
                    m=1                
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
                if(mask_per_sample==False):
                    m=1                
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
                if(mask_per_sample==False):
                    m=1                
                inds=np.random.rand(n,m).argsort(axis=1)[:,0:nmask]
                data[np.arange(len(data))[:, None], inds] = my_mask
                return data
            
            pool = multiprocessing.pool.ThreadPool(par_mask_proc)
            result = pool.map(mask_worker,chunks(mydata,par_mask_proc))
            pool.close()
            pool.join()    
            result = [val for sublist in result for val in sublist]
            mydata = np.array(result)
            
        elif(par_mask_method=="ray"):
            @ray.remote
            def mask_worker(data):
                n=len(data)
                m=len(data[0])
                if(mask_per_sample==False):
                    m=1
                inds=np.random.rand(n,m).argsort(axis=1)[:,0:nmask]
                #fails if we try to overwrite, solution bellow
                data.setflags(write=1)                
                data[np.arange(len(data))[:, None], inds] = my_mask               

            futures = [mask_worker.remote(i) for i in chunks(mydata,par_mask_proc)]
            result = ray.get(futures)
            result = np.array(result)
            mydata = result
        else:
            m=len(mydata[0])
            if(mask_per_sample==False):
                s=1         
            inds=np.random.rand(s,m).argsort(axis=1)[:,0:nmask]
            #slower
            #inds=np.stack([np.random.choice(np.arange(m),size=nmask,replace=False) for i in range(s)])
            mydata[np.arange(s)[:, None], inds] = my_mask
    else:
        #REAAAAAAALLY SLOOOOOOW, not using this anymore
        j = 0
        while j < len(mydata):
            #redefines which variants will be masked for every new sample
            if(mask_per_sample==True or j==0):
                maskindex = random.sample(range(0, len(mydata[0]-1)), nmask) 
            for i in maskindex:
                mydata[j][i]=my_mask
            j=j+1

    if(verbose>0):
        print("Masked shape:", mydata.shape, flush=False)
    return mydata
    
def variable_summaries(var):
    #Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def calculate_rho_hat(activations, act_val):

    rho_hat = tf.reduce_mean(activations,0) #RR sometimes returns Inf in KL function, caused by division by zero, fixed wih logfun()

    if(act_val.startswith("tanh") or act_val.startswith("relu") or act_val.startswith("softplus")):
        rho_hat = tf.add(rho_hat,tf.abs(tf.reduce_min(rho_hat))) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl
        rho_hat = tf.div_no_nan(rho_hat,tf.reduce_max(rho_hat)) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl

    #avoid nan
    rho_hat = tf.clip_by_value(rho_hat, eps, one-eps)
    return rho_hat

#Kullback-Leibler divergence equation (KL divergence)
#The result of this equation is added to the loss function result as an additional penalty to the loss based on sparsity
def KL_Div(rho, rho_hat):
    #RR I just simplified the KL divergence equation according to the book:
    #RR "Hands-On Machine Learning with Scikit-Learn and TensorFlow" by Aurélien Géron
    #RR Example source code here https://github.com/zhiweiuu/sparse-autoencoder-tensorflow/blob/master/SparseAutoEncoder.py
    #RR KL2 is the classic sparsity implementation, source reference: https://github.com/elykcoldster/sparse_autoencoder/blob/master/mnist_sae.py
    #https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
    #default suggested beta and rho for denoising sparse AE: 0.1, 0.01: https://github.com/jbregli/Stacked_auto_encoders/blob/master/CPU_version/AutoEncoder.py
    rho = tf.cast(rho, tf_precision)
    rho_hat = tf.cast(rho_hat, tf_precision)

    #https://gist.github.com/morphogencc/c4141ec923310d71e2ea62f5ee227f95
    KL_loss = rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)
    if(average_loss==True):
        KL_loss = tf.reduce_mean(KL_loss)
    else:
        KL_loss = tf.reduce_sum(KL_loss)

    return KL_loss

def calculate_pt(y_pred, y_true):

    y_pred = tf.cast(y_pred, tf_precision)
    y_true = tf.cast(y_true,tf_precision)

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

    #ref:https://github.com/unsky/focal-loss/blob/master/focal_loss.py
    pt_1 = tf.clip_by_value(pt_1, eps, one) #avoid log(0) that returns inf

    #if value is zero in y_true, than take value from y_pred, otherwise, write zeros
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_0 = tf.clip_by_value(pt_0, eps, one)
        
    return pt_0, pt_1

def calculate_CE(y_pred, y_true):

    pt_0, pt_1 = calculate_pt(y_pred, y_true)
    one_eps = tf.cast(1.0+1e-6, tf_precision)

    CE_1 = tf.multiply(n1,tf.log(pt_1))
    CE_0 = tf.multiply(n1,tf.log(tf.subtract(one_eps,pt_0)))

    return CE_0, CE_1

def cross_entropy(y_pred, y_true):

    pt_0, pt_1 = calculate_pt(y_pred, y_true)

    CE_0, CE_1 =  calculate_CE(y_pred, y_true)

    CE = tf.add(CE_1, CE_0)

    if(average_loss==True and loss_type=='CE'):
        CE = tf.reduce_mean(CE, name='reconstruction_loss')
    else:
        CE = tf.reduce_sum(CE, name='reconstruction_loss')

    return CE

def calculate_alpha(y_true):

    alpha = tf.cast(freq_all_var, tf_precision)

    #keep alpha returning duplicated in case we want to do any specific transformation
    return alpha, alpha

def weighted_cross_entropy(y_pred, y_true):

    CE_0, CE_1 =  calculate_CE(y_pred, y_true)

    alpha_0, alpha_1 = calculate_alpha(y_true)

    WCE_per_var_1 = tf.multiply(CE_1, alpha_1)
    WCE_per_var_0 = tf.multiply(CE_0, alpha_0)

    WCE_1 = tf.reduce_sum(WCE_per_var_1)
    WCE_0 = tf.reduce_sum(WCE_per_var_0)

    if(average_loss==True and loss_type=='CE'):
        WCE = tf.reduce_mean(tf.add(WCE_1, WCE_0), axis=0, name='reconstruction_loss')
    else:
        WCE = tf.add(WCE_1, WCE_0, name='reconstruction_loss')

    return WCE


def calculate_gamma(y_pred, y_true, my_gamma):

    pt_0, pt_1 = calculate_pt(y_pred, y_true)

    gamma_0 = tf.pow(pt_0, my_gamma)
    gamma_1 = tf.pow(tf.subtract(one, pt_1), my_gamma)

    return gamma_0, gamma_1

#ref: https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
def focal_loss(y_pred, y_true):
    
    CE_0, CE_1 =  calculate_CE(y_pred, y_true)

    alpha_0, alpha_1 = calculate_alpha(y_true)

    FL_per_var_1_a = CE_1[:,1::2]
    FL_per_var_0_a = CE_0[:,1::2]
    FL_per_var_1_r = CE_1[:,0::2]
    FL_per_var_0_r = CE_0[:,0::2]

    my_gamma=tf.Variable(gamma, name="gamma")
    tf.add_to_collection('gamma', my_gamma)

    print("gamma",my_gamma)

    gamma_0, gamma_1 = calculate_gamma(y_pred[:,1::2], y_true[:,1::2], my_gamma)

    FL_per_var_1_a = tf.multiply(gamma_1, FL_per_var_1_a)
    FL_per_var_0_a = tf.multiply(gamma_0, FL_per_var_0_a)

    gamma_0, gamma_1 = calculate_gamma(y_pred[:,0::2], y_true[:,0::2], my_gamma)

    FL_per_var_1_r = tf.multiply(gamma_1, FL_per_var_1_r)
    FL_per_var_0_r = tf.multiply(gamma_0, FL_per_var_0_r)

    FL_per_var_r = tf.add(FL_per_var_0_r, FL_per_var_1_r)
    FL_per_var_a = tf.add(FL_per_var_0_a, FL_per_var_1_a)

    if(disable_alpha==False):
        #extract and reweight alternative allele
        FL_per_var_a = tf.multiply(FL_per_var_a, alpha_1[1::2])
        #extract and reweight reference allele
        FL_per_var_r = tf.multiply(FL_per_var_r, alpha_1[0::2])

    FL = tf.concat([FL_per_var_r,FL_per_var_a], 1)

    if(average_loss==True):
        FL = tf.reduce_mean(FL,  name='reconstruction_loss')
    else:
        FL = tf.reduce_sum(FL, name='reconstruction_loss')

    return FL

def f1_score(y_pred, y_true, sess):
    
    f1s = [0, 0, 0]
    two = tf.cast(2.0, tf_precision)

    y_true = tf.cast(y_true, tf_precision)
    y_pred = tf.cast(y_pred, tf_precision)
    
    y_true = tf.clip_by_value(y_true, 0.0, 1.0) #in case the input encoding is [0,1,2]
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(tf.multiply(y_pred, y_true), axis=axis)
        FP = tf.count_nonzero(tf.multiply(y_pred, tf.subtract(y_true,1.0)), axis=axis)
        FN = tf.count_nonzero(tf.multiply(tf.subtract(y_pred,1.0),y_true), axis=axis)
        
        TP = tf.cast(TP, tf_precision)
        FP = tf.cast(FP, tf_precision)
        FN = tf.cast(FN, tf_precision)
        
        TP = tf.add(TP, eps)
        
        precision = tf.div_no_nan(TP, tf.add(TP, FP))
        recall = tf.div_no_nan(TP, tf.add(TP, FN))
        top = tf.multiply(precision, recall)
        bottom = tf.add(precision, recall)
        f1 = tf.multiply(two, tf.div_no_nan(top,bottom))

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights = tf.div_no_nan(weights,tf.reduce_sum(weights))

    f1s[2] = tf.reduce_sum(tf.multiply(f1, weights))

    micro, macro, weighted = sess.run(f1s)
    
    return micro, macro, weighted

def pearson_r2(x_in, y_in):

    results=dict()
    r2_results_l = []
    p_results = []

    x = np.copy(x_in)
    y = np.copy(y_in)

    #per sample
    x_sum = np.sum(x, axis=1)
    y_sum = np.sum(y, axis=1)
    xy_sum = np.sum(np.multiply(x,y), axis=1)
    x_squared_sum = np.sum(np.power(x,2), axis=1)
    y_squared_sum = np.sum(np.power(y,2), axis=1)
    N=len(y[0])

    num=np.subtract(np.multiply(xy_sum, N), np.multiply(x_sum, y_sum) )
    den=np.multiply(x_squared_sum, N)
    den=np.subtract(den, np.power(x_sum,2))
    den2=np.multiply(y_squared_sum, N)
    den2=np.subtract(den2, np.power(y_sum,2))
    den=np.sqrt(np.multiply(den, den2))

    eps=1e-6

    num = np.add(eps,num)
    den = np.add(eps,den)

    r2_per_sample=np.divide(num,den)
    r2_per_sample=np.power(r2_per_sample,2)
    r2_per_sample=np.round(r2_per_sample, decimals=3)

    #per variant
    x_sum = np.sum(x, axis=0)
    y_sum = np.sum(y, axis=0)
    xy_sum = np.sum(np.multiply(x,y), axis=0)
    x_squared_sum = np.sum(np.power(x,2), axis=0)
    y_squared_sum = np.sum(np.power(y,2), axis=0)

    results['r2_per_sample']=r2_per_sample

    results['x_sum']=x_sum
    results['y_sum']=y_sum
    results['xy_sum']=xy_sum
    results['x_squared_sum']=x_squared_sum
    results['y_squared_sum']=y_squared_sum

    #results['mean_x_per_sample']=mean_x
    #results['mean_y_per_sample']=mean_y
    #results['var_y_per_sample']=var_y
    #results['var_x_per_sample']=var_x
    #results['covar']=covar
    results['N']=len(y)

    return results    
    

#drop out function will exclude samples from our input data
#keep probability (kp) or keep rate will determine how many samples will remain after randomly droping out samples from the input
def dropout(input, name, kp):
    with tf.name_scope(name):
        out = tf.nn.dropout(input, kp)
    return out
    # call function like this, p1 is input, name is layer name, and keep rate doesnt need explanation,  
    # do1 = dropout(p1, name='do1', keep_rate=0.75)

    #A value of 1.0 means that dropout will not be used.
    #TensorFlow documentation https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#dropout

#Example adapted and modified from 
#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

#Encodes a Hidden layer
def encoder(x, func, l1_val, l2_val, weights, biases, units_num, keep_rate): #RR added keep_rate
    x=tf.cast(x, tf_precision)
    
    print("Setting up encoder/decoder.")
    if(l2_val==0):
        regularizer = tf.contrib.layers.l1_regularizer(l1_val)
    else:
        regularizer = tf.contrib.layers.l1_l2_regularizer(l1_val,l2_val)

    print("keep_rate:", keep_rate)
    #dropout
    if keep_rate != 1: ##RR added dropout
            x = dropout(x, 'x', keep_rate) ##RR added dropout

    if(func.startswith('sigmoid')):
        print('Encoder Activation function: sigmoid')
        
        #tf.nn.sigmoid computes sigmoid of x element-wise.
        #Specifically, y = 1 / (1 + exp(-x))
        #tf.matmul multiply output of input_layer with a weight matrix and add biases
        #tf.matmul Multiplies matrix a by matrix b, producing a * b
        #If one or both of the matrices contain a lot of zeros, a more efficient multiplication algorithm can be used by setting the corresponding a_is_sparse or b_is_sparse flag to True. These are False by default.
        #tf.add will sum the result from tf.matmul (input*weights) to the biases
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

    elif(func.startswith('tanh')):
        print('Encoder Activation function: tanh')
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

    elif(func.startswith('relu')):
        print('Encoder Activation function: relu')
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))

    elif(func.startswith('softplus')):
        print('Encoder Activation function: softplus')
        layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

    #This layer implements the operation: 
    #outputs = activation(inputs * weights + bias) 
    #where activation is the activation function passed as the activation argument, defined inprevious line
    #The function is applied after the linear transformation of inputs, not W*activation(inputs)
    #Otherwise the function would not allow nonlinearities
    layer_1 = tf.layers.dense(layer_1, units=units_num, kernel_regularizer= regularizer)


    return layer_1

def decoder(x, func, weights, biases):
    
    x = tf.cast(x, tf_precision)
    
    if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
        entropy_loss = True
    else:
        entropy_loss = False
        
    if(func.endswith('sigmoid')):
        print('Decoder Activation function: sigmoid')
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']), name="y_pred")

    elif(func.endswith('tanh')):
        print('Decoder Activation function: tanh')
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))

    elif(func.endswith('relu')):
        print('Decoder Activation function: relu')
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))

    elif(func.endswith('softplus')):
        print('Decoder Activation function: softplus')
        layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))

    elif(func.endswith('none')):
        print('Decoder Activation function: none')
        layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    if(not func.endswith("sigmoid")):
        layer_1 = tf.add(layer_1,tf.abs(tf.reduce_min(layer_1))) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl
        layer_1 = tf.div_no_nan(layer_1,tf.reduce_max(layer_1), name="y_pred") # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl

    return layer_1


def mean_empirical_r2(x,y):
    #This calculates exactly the same r2 as the empirical r2hat from minimac3 and minimac4
    #The estimated r2hat is different
    j=0
    mean_correl = 0
    correls = []
    while j < len(y[0]):
        getter = operator.itemgetter([j+1])
        x_genotypes = list(map(list, map(getter, np.copy(x))))
        y_genotypes = list(map(list, map(getter, np.copy(y))))
        x_genotypes = list(np.array(x_genotypes).flat)
        y_genotypes = list(np.array(y_genotypes).flat)
        correl = linregress(x_genotypes, y_genotypes)[2]
        mean_correl += correl/len(y[0])
        j+=2
        correls.append(mean_correl)
    return mean_correl, correls

def filter_by_MAF(x,y, MAFs, threshold1=0, threshold2=1):
    
    colsum=np.sum(y, axis=0)
    indexes_to_keep = []
    i = 0
    j = 0
    k = 0   
    if(verbose>0):
        print("COOLSUM", len(colsum), ":", colsum[0:13])

    while i < len(MAFs):
        if(MAFs[i]>threshold1 and MAFs[i]<=threshold2):
            if(colsum[k]!=0 or colsum[k+1]!=0):
                indexes_to_keep.append(k)
                indexes_to_keep.append(k+1)
            else:
                print("WARNING!!!!! INDEX", i, "HAS good MAF but colsum is zero")            
        i += 1
        k += 2
    if(verbose>0):
        print("FILTER BY MAF INDEXES TO KEEP:", len(indexes_to_keep), indexes_to_keep)

    getter = operator.itemgetter(indexes_to_keep)

    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
    
    return filtered_data_x, filtered_data_y

def accuracy_maf_threshold(sess, x, y, MAFs, threshold1=0, threshold2=1):

    filtered_data_x, filtered_data_y = filter_by_MAF(x,y, MAFs, threshold1, threshold2)
    
    correct_prediction = np.equal( np.round( filtered_data_x ), np.round( filtered_data_y ) )
    accuracy_per_marker = np.mean(correct_prediction.astype(float), 0)
    accuracy = np.mean(accuracy_per_marker)

    return accuracy, accuracy_per_marker

def MSE_maf_threshold(sess, x, y, MAFs, threshold1=0, threshold2=1):
    
    filtered_data_x, filtered_data_y = filter_by_MAF(x,y, MAFs, threshold1, threshold2)
    
    MSE_per_marker = np.mean(np.square( np.subtract( np.round( filtered_data_x ), np.round( filtered_data_y ) ) ), 0 ) 
    MSE = np.mean( MSE_per_marker )
    
    return MSE, MSE_per_marker

def accuracy_maf_threshold_global(sess, x, y, indexes_to_keep):
    
    getter = operator.itemgetter(indexes_to_keep)
    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
       
    correct_prediction = np.equal( np.round( filtered_data_x ), np.round( filtered_data_y ) )
    accuracy_per_marker = np.mean(correct_prediction.astype(float), 0)
    accuracy = np.mean(accuracy_per_marker)


    return accuracy, accuracy_per_marker

def MSE_maf_threshold_global(sess, x, y, indexes_to_keep):
    
    getter = operator.itemgetter(indexes_to_keep)
    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
    
    MSE_per_marker = np.mean(np.square( np.subtract( np.round( filtered_data_x ), np.round( filtered_data_y ) ) ), 0 ) 
    MSE = np.mean( MSE_per_marker )

    return MSE, MSE_per_marker

def flatten_data_np(x):
    x = np.reshape(x, (x.shape[0],-1))
    return x

def define_weights(ni,nh,no,last_layer=False):
    w = {
        'encoder_h1': tf.Variable(tf.random_normal([ni, nh], dtype=tf_precision), name="w_encoder_h1"),
    }
    if(last_layer==True):
        w['decoder_h1']=tf.Variable(tf.random_normal([nh, no], dtype=tf_precision), name="w_decoder_h1")
    return w

def define_biases(nh,no,last_layer=False):
    
    b = {
        'encoder_b1': tf.Variable(tf.random_normal([nh], dtype=tf_precision), name="b_encoder_b1"),
    }
    if(last_layer==True):
        b['decoder_b1']=tf.Variable(tf.random_normal([no], dtype=tf_precision), name="b_decoder_b1")
    return b
#Code modified from example
#https://stackoverflow.com/questions/44367010/python-tensor-flow-relu-not-learning-in-autoencoder-task
def run_autoencoder(learning_rate, training_epochs, l1_val, l2_val, act_val, beta, rho, data_obs):

    prep_start = timeit.default_timer()
    beta = tf.cast(beta, tf_precision)
    
    print("Running autoencoder.")
    
    print("Input data shape:")
    print(data_obs.shape)
    
    original_shape = data_obs.shape
    
    batch_size = int(round(len(data_obs)/split_size)) # size of training objects split the dataset in 10 parts
    print("Batch size:",batch_size)
    
    display_step = 1        

    # define layer size
    if(len(data_obs.shape) == 3): #if alleles are represented as either vectors of counts or presence/absense
        n_input = len(data_obs[0])*len(data_obs[0][0])
        if(left_buffer>0):
            n_left_buffer = left_buffer*len(data_obs[0][0])
        if(right_buffer>0):
            n_right_buffer = right_buffer*len(data_obs[0][0])
    else: #if doing one hot encoding, experimental
        n_input = len(data_obs[0])*3     # input features N_variants*subfeatures
        if(left_buffer>0):
            n_left_buffer = left_buffer*3
        if(right_buffer>0):
            n_right_buffer = right_buffer*3

    global hsize

    #initially set output layer size equal to input layer size
    n_output = n_input
    
    #if user provides left or right number, the buffer size is subtracted from the output layer size
    if(left_buffer>0):
        n_output = n_output-n_left_buffer
    if(right_buffer>0):
        n_output = n_output-n_right_buffer

    #enable support for different output layer sizes
    if(n_input!=n_output):
        o_indexes = list(range(n_left_buffer, n_left_buffer+n_output))    
        o_getter = operator.itemgetter(o_indexes)

    #all hidden layer calculations now are based on the output layer size as reference, not input    
    if(hsize=="sqrt"):
        n_hidden=[int(round(np.sqrt(n_output)))] #if hsize equal 'sqrt' hidden layer size equal to square root of number of input nodes
    else:
        hsize=convert_to_float(hsize)
        if(type(hsize) != type([])):
            hsize = [hsize]
        # hidden layer for encoder, equal to input number of features multiplied by a hidden size ratio
        n_hidden = [int(round(n_output*i)) for i in hsize]
    
    #if there are less hidden layer size values than number of hidden layer, replicate the first hidden layer size into the additional layers
    while(len(n_hidden)<NH):
        n_hidden.append(n_hidden[0])

    print("Input data shape after coding variables:")
    print(n_input)
    
    print("Network size per layer:")
    #print(n_input, n_hidden_1, n_output)
    print(n_input, n_hidden, n_output)
    
    # Input placeholders
    #with tf.name_scope('input'):
        #tf input
    X = tf.placeholder("float", [None, n_input], name="X")
    #now output layer allows different sizes
    Y = tf.placeholder("float", [None, n_output], name="Y")
    #Y = tf.placeholder("float", [None, n_input], name="Y")
            
    #As parameters of a statistical model, weights and biases are learned or estimated by minimizing a loss function that depends on our data. 
    #We will initialize them here, their values will be set during the learning process

    n_hidden = [n_input] + n_hidden
    weights_per_layer = []
    biases_per_layer = []    
    last_layer=False
    for i in range(NH):
        if(i==(NH-1)):
            last_layer=True
        with tf.name_scope('biases'):
            biases_per_layer.append(define_biases(n_hidden[i+1], n_output,last_layer))
            variable_summaries(biases_per_layer[i]['encoder_b1'])                   
        with tf.name_scope('weights'):
            weights_per_layer.append(define_weights(n_hidden[i],n_hidden[i+1],n_output,last_layer))
            variable_summaries(weights_per_layer[i]['encoder_h1'])            

    encoder_operators = [X]
    for i in range(NH):
        with tf.name_scope('Wx_plus_b'):
            encoder_operators.append(encoder(encoder_operators[i], act_val, l1_val, l2_val, weights_per_layer[i], biases_per_layer[i], n_hidden[i+1], keep_rate[i]))
            tf.summary.histogram('activations', encoder_operators[i+1])

    y_pred = decoder(encoder_operators[-1], act_val, weights_per_layer[NH-1], biases_per_layer[NH-1])            
    print(encoder_operators[-1])
    tf.summary.histogram('activations', encoder_operators[1])

    y_pred = tf.identity(y_pred, name="y_pred")

    y_true = Y

    rho_hat = []
    #apply sparsity cost just to first layer if all_sparse==False
    rho_hat.append(calculate_rho_hat(encoder_operators[1], act_val))
    if(all_sparse==True):
        for i in range(2,NH+1):
           rho_hat.append(calculate_rho_hat(encoder_operators[i], act_val))

    with tf.name_scope('sparsity'):
        sparsity_loss = KL_Div(rho, rho_hat[0])
        sparsity_loss = tf.cast(sparsity_loss, tf_precision)
        if(all_sparse==True):
            for i in range(1,NH):
                sparsity_loss = tf.add(sparsity_loss, tf.cast(KL_Div(rho, rho_hat[i]), tf_precision))
            sparsity_loss = tf.div_no_nan(sparsity_loss, tf.cast(NH, tf_precision))
        sparsity_loss = tf.cast(sparsity_loss, tf_precision, name="sparsity_loss") #RR KL divergence, clip to avoid Inf or div by zero

    tf.summary.scalar('sparsity_loss', sparsity_loss)

    # define cost function, optimizers
    with tf.name_scope('loss'):

        if(loss_type=="MSE"):
            y_true = tf.cast(y_true, tf_precision)
            reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred)), name="reconstruction_loss") #RR simplified the code bellow
        elif(loss_type=="CE"):
            reconstruction_loss = cross_entropy(y_pred, y_true)
        elif(loss_type=="WCE"):
            reconstruction_loss = weighted_cross_entropy(y_pred, y_true)
        elif(loss_type=="FL"):
            reconstruction_loss = focal_loss(y_pred, y_true)
            if(verbose>0):
                mygamma_0, mygamma_1 = calculate_gamma(y_pred, y_true,tf.cast(gamma, tf_precision))
                ce0, ce1 = calculate_CE(y_pred, y_true)
                pt0, pt1 = calculate_pt(y_pred, y_true)
                wce = weighted_cross_entropy(y_pred, y_true)
        else:
            y_true = tf.cast(y_true, tf_precision)            
            reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred)), name="reconstruction_loss") #RR 

        cost = tf.reduce_mean(tf.add(reconstruction_loss,tf.multiply(beta, sparsity_loss)), name = "cost") #RR simplified equation
        
    tf.summary.scalar('reconstruction_loss_MSE', reconstruction_loss)
    tf.summary.scalar("final_cost", cost)

    correct_prediction = 0
    
    y_true = tf.cast(y_true, tf_precision)
    correct_prediction = tf.equal( tf.round( y_pred ), tf.round( y_true ) )
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf_precision), name="accuracy")
    cost_accuracy = tf.add((1-accuracy), tf.multiply(beta, sparsity_loss), name="cost_accuracy")
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('cost_accuracy', cost_accuracy)
    
    #ref: https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b
    if(recovery_mode=="False"):
        with tf.name_scope('train'):
            if(optimizer_type=="RMSProp"):
                #This is a variant Adadelta that serves the same purpose, dynamic decay of a learning rate multiplier
                optimizer = tf.train.RMSPropOptimizer(learning_rate, name="optimizer")
            elif(optimizer_type=="GradientDescent"):
                #Typical gradient descent with fixed learning rate
                optimizer = tf.train.GradientDescentOptimizer(learning_rate, name="optimizer")
            elif(optimizer_type=="Adam"):
                #Adaptive momentum in addition to the Adadelta features
                optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer")
            elif(optimizer_type=="Adadelta"):
                #Linearly decaying learning rate
                optimizer = tf.train.AdadeltaOptimizer(learning_rate, name="optimizer")
            elif(optimizer_type=="Adagrad"):
                #Adaptively scales the learning rate for each dimension
                optimizer = tf.train.AdagradOptimizer(learning_rate, name="optimizer")

            optimizer = optimizer.minimize(cost)

        print("Optimizer:", optimizer)

    # initialize variables
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    start = timeit.default_timer()
    r_report_time = 0
    mask_time = 0
    time_metrics = 0
    gd_time = 0
    # run autoencoder .........
    if(recovery_mode=="True"):
        tf.reset_default_graph()

#    with tf.compat.v1.Session(config=config) as sess:
    with tf.Session(config=config) as sess:
        
        if(save_summaries==True):
            
            train_writer = tf.summary.FileWriter('./train', sess.graph)
            valid_writer = tf.summary.FileWriter('./valid')

            low_MAF_summary=tf.Summary()

            high_MAF_summary=tf.Summary()
        
            run_metadata = tf.RunMetadata()

        merged = tf.summary.merge_all()

        if(recovery_mode=="True"):
            print("Restoring model from checkpoint")
            #recover model from checkpoint
            meta_path = model_path + '.meta'
            saver = tf.train.import_meta_graph(meta_path)
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            #saver.restore(sess, tf.train.latest_checkpoint('HRC.r1-1.EGA.GRCh37.chr22.haplotypes.17274081-17382360.vcf.VMV1_model_round1/'))
            graph = sess.graph

            global_name=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
            local_name=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=tf.get_variable_scope().name)
            trainable_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            if(verbose>0):
                print("Global collection:", global_name)
                print("Local collection:", local_name)
                print("Trainable:", trainable_vars)

            #optimizer = graph.get_tensor_by_name("optimizer:0")

            a = [n.name for n in graph.as_graph_def().node]
            if("train/optimizer" in a):
                 optimizer = graph.get_operation_by_name( "train/optimizer" )
                 print("Restored", "train/optimizer")
            elif("train/optimizer/train/optimizer/-apply" in a):
                 optimizer = graph.get_operation_by_name( "train/optimizer/train/optimizer/-apply" )
                 print("Restored", "train/optimizer/-apply")
            else:
                 print("ERROR OPTIMIZER NOT FOUND IN GRAPH. Nodes available: ", a)

            X = graph.get_tensor_by_name("X:0")
            Y = graph.get_tensor_by_name("Y:0")
            #X = "X:0"
            #Y = "Y:0"
            cost = graph.get_tensor_by_name("loss/cost:0")
            #cost = "cost:0"
            reconstruction_loss = graph.get_tensor_by_name("loss/reconstruction_loss:0")
            #reconstruction_loss = "reconstruction_loss:0"
            #sparsity_loss = "sparsity_loss:0"            

            if("sparsity/sparsity_loss" in a):
                sparsity_loss = graph.get_tensor_by_name("sparsity/sparsity_loss:0")
                print("Restored", "sparsity/sparsity_loss")
            elif("sparsity/Add_4" in a):
                sparsity_loss = graph.get_tensor_by_name("sparsity/Add_4:0")
                print("Restored", "sparsity_loss")
            else:
                print("ERROR OPTIMIZER NOT FOUND IN GRAPH. Nodes available: ", a)

            accuracy = graph.get_tensor_by_name("accuracy:0")

            #accuracy = "accuracy:0"
            #cost_accuracy = "cost_accuracy:0"
            cost_accuracy = graph.get_tensor_by_name("cost_accuracy:0")
            y_pred = graph.get_tensor_by_name("y_pred:0")
           
            v = [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
            if("loss/gamma:0" in v):
                new_gamma = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == 'loss/gamma:0'][0]
                print("Old gamma value", sess.run(new_gamma))
                sess.run(tf.assign(new_gamma, gamma))
                gamma_check = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == 'loss/gamma:0'][0]
                print("Checking if gamma value really changed:", sess.run(gamma_check))
            else:
                print("WARNING!!! Gamma variable not found. Searched for name loss/gamma:0.")

            tf.summary.scalar('reconstruction_loss_MSE', reconstruction_loss)
            tf.summary.scalar("final_cost", cost)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('cost_accuracy', cost_accuracy)
            tf.summary.scalar('sparsity_loss', sparsity_loss)

            decoder_op = y_pred

            if(save_summaries==True):

                merged = tf.summary.merge_all()
            #y_pred = "y_pred:0"
            print("Model restored")
        else:    
            sess.run(init)

        prep_stop = timeit.default_timer()
        prep_time = prep_stop-prep_start
        
        mean_cost = 0
        mean_rloss = 0
        mean_sloss = 0
        mean_acc = 0
        mean_cacc = 0   
        mean_F1 = [0,0,0]
        mean_acc_t = [0,0,0,0,0,0,0]
        my_acc_t = [0,0,0,0,0,0,0]
      
        mask_start = timeit.default_timer()
            
        mask_rate = initial_masking_rate
        indexes = list(range(len(data_obs)))        
                      
        if(fixed_masking == True and disable_masking==False): #mask all data only once before the whole training procedure         
            if(initial_masking_rate > 0):
                mask_rate = initial_masking_rate
            else:
                mask_rate = 0.9 #set default if no value is provided
            data_masked = mask_data_per_sample_parallel([0],np.copy(data_obs),mask_rate)

            mask_stop = timeit.default_timer()
            print("Time to run masking: ", mask_stop-mask_start)
            mask_time += mast_stop-mask_start

            if(disable_masking==False):
                data_masked = flatten_data_np(data_masked)
                
            data_obs = flatten_data_np(data_obs)

            if(n_input!=n_output):
                data_obs = list(map(list, map(o_getter, np.copy(data_obs))))

            print(data_masked.shape)
            
        global mask_increase

        if(fixed_masking == True and disable_masking==False):
            train_x = np.copy(data_masked)
            del data_masked
        else:
            train_x = np.copy(data_obs)                    
        train_y = np.copy(data_obs)

        total_batch = int(train_x.shape[0] / batch_size)
        print(train_x.shape)

        if(fixed_masking_rate==True):
            mask_rate = initial_masking_rate

        time_epochs = 0
        flat_time = 0
        shuf_time = 0
        rest_time = 0
        save_time = 0

        epoch=0
        cycle_count = -1
        previous_window_avg_cost = 0
        current_window_avg_cost = 0
        avg_cost = 0
        val_avg_cost = 0
        time_to_stop=False
        
        if(save_summaries==True):
            if(verbose>1):
                train_args = [merged,optimizer, cost, accuracy, reconstruction_loss, sparsity_loss, rho_hat, mygamma_0, mygamma_1, ce0, ce1,pt0, pt1, wce, y_pred]
            else:
                train_args = [merged,optimizer, cost, accuracy, reconstruction_loss, y_pred]
        else:
            if(verbose>1):
                train_args = [optimizer, cost, accuracy, reconstruction_loss, sparsity_loss, rho_hat, mygamma_0, mygamma_1, ce0, ce1, pt0, pt1, wce, y_pred]
            else:
                if(calculate_r2_per_epoch==True):
                    train_args = [optimizer, cost, accuracy, reconstruction_loss, y_pred]
                else:
                    train_args = [optimizer, cost, accuracy, reconstruction_loss]
                    
        if(calculate_r2_per_epoch==True):
            if(save_summaries==True):
                pred_args = [merged,accuracy, cost, y_pred, encoder_op, reconstruction_loss]
            else:
                pred_args = [accuracy, cost, reconstruction_loss, y_pred]
        else:
            if(save_summaries==True):
                pred_args = [merged,accuracy, cost, encoder_op, reconstruction_loss]
            else:
                pred_args = [accuracy, cost, reconstruction_loss]

        #EPOCH STARTS HERE
        for iepoch in range(training_epochs+1):

            if(recovery_mode=="True"):
                epoch=iepoch+resuming_step
            else:
                epoch=iepoch

            start_epochs = timeit.default_timer()
            mask_start = timeit.default_timer()

            if(fixed_masking_rate==False and mask_rate<maximum_masking_rate and cycle_count==repeat_cycles):
                if(mask_increase>0):
                    mask_rate += mask_increase
                else:
                    mask_rate += initial_masking_rate
                mask_rate=np.round(mask_rate,6)

            if(mask_rate>maximum_masking_rate):
                mask_rate=maximum_masking_rate

            if(cycle_count==repeat_cycles):
                cycle_count = 0
            else:
                cycle_count += 1

            #make new masking on every new iteration
            if( (epoch>=0 and do_parallel_gd_and_mask==False) or epoch==0):
                if(do_parallel_numpy_per_cycle == False):
                    data_masked = mask_data_per_sample_parallel([0], np.copy(data_obs), mask_rate)
                if(do_parallel_numpy_per_cycle==True):
                    if(cycle_count==0):
                        m_result, pool = submit_mask_tasks(np.copy(data_obs), mask_rate)
                        data_masked_replicates = retrieve_cycle_results(m_result,pool)
                        
                    data_masked = data_masked_replicates[cycle_count]
                
                if(validate_after_epoch==True or (validate_after_cycle==True and cycle_count==repeat_cycles) ):
                    data_val_masked = mask_data_per_sample_parallel([0], np.copy(data_obs),maximum_masking_rate)
                    val_x = data_val_masked
                if(verbose>0):
                    print("Mask done for epoch",iepoch,"Result length:",len(data_masked_replicates), "Data masked shape:", data_masked.shape)
                    
            train_x = np.copy(data_masked)
            train_y = np.copy(data_obs)

            mask_stop = timeit.default_timer()
            mask_time += mask_stop-mask_start

            flat_start = timeit.default_timer()
            #after masking, flatten data
            train_x = flatten_data_np(train_x)
            train_y = flatten_data_np(train_y)
            
            if(validate_after_epoch==True or (validate_after_cycle==True and cycle_count==repeat_cycles) ):
                val_x = flatten_data_np(val_x)

            if(n_input!=n_output):
                train_y = list(map(list, map(o_getter, np.copy(train_y))))

            flat_stop = timeit.default_timer()
            flat_time += flat_stop-flat_start

            shuf_start = timeit.default_timer()
            if(shuffle==True):
                randomize = np.random.rand(len(train_x)).argsort()
                train_x = train_x[randomize]
                train_y = train_y[randomize]
                if(validate_after_epoch==True or (validate_after_cycle==True and cycle_count==repeat_cycles) ):
                    val_x = np.asarray(val_x)
                    val_x = val_x[randomize]

            shuf_stop = timeit.default_timer()
            shuf_time += shuf_stop-shuf_start
                   
            avg_r2 = 0
            epoch_cost = 0
            avg_a = 0
            avg_rl = 0
            
            if(calculate_r2_per_epoch==False):
                avg_r2="DISABLED"
            if(calculate_acc_per_epoch==False):
                avg_a="DISABLED"

            val_avg_a = 0
            val_avg_rl = 0
            val_avg_r2 = 0
            val_epoch_cost = 0
            
            if(do_parallel_gd_and_mask==True and iepoch>0):
                if(verbose>0):
                    print("Running parallel gd and mask")
                
                if(do_parallel_numpy_per_cycle==False or cycle_count==0):
                
                    mask_start = timeit.default_timer()
                    m_result, pool = submit_mask_tasks(data_obs, mask_rate)
                    mask_stop = timeit.default_timer()
                    mask_time += mask_stop-mask_start
                    
                    gd_start = timeit.default_timer()
                    my_result_list = run_epoch_worker(sess, X,Y,data_masked, data_obs, train_args, batch_size)
                    gd_stop = timeit.default_timer()
                    gd_time += gd_stop-gd_start
                    
                    mask_start = timeit.default_timer()
                    data_masked_replicates = retrieve_cycle_results(m_result,pool)
                    mask_stop = timeit.default_timer()
                    mask_time += mask_stop-mask_start
                
                mask_start = timeit.default_timer()
                
                if(cycle_count>0 and do_parallel_numpy_per_cycle==False):
                    data_masked = data_masked_replicates[0]
                else:
                    data_masked = data_masked_replicates[cycle_count]
                    
                mask_stop = timeit.default_timer()
                mask_time += mask_stop-mask_start
                
                if(do_parallel_numpy_per_cycle==True and cycle_count>0):
                    gd_start = timeit.default_timer()
                    my_result_list = run_epoch_worker(sess, X, Y, data_masked, data_obs, train_args, batch_size)
                    gd_stop = timeit.default_timer()
                    gd_time += gd_stop-gd_start
                               
                if(verbose>0):
                    print("Parallel gd and mask done")                

            for i in range(total_batch):
                
                rest_start = timeit.default_timer()
                
                if(do_parallel_gd_and_mask==False or iepoch==0):                
                    batch_x = train_x[i*batch_size:(i+1)*batch_size]
                    batch_y = train_y[i*batch_size:(i+1)*batch_size]                   

                if(validate_after_epoch==True or (validate_after_cycle==True and cycle_count==repeat_cycles) ):
                    batch_val_x = val_x[i*batch_size:(i+1)*batch_size]
                    
                rest_stop = timeit.default_timer()
                rest_time += rest_stop-rest_start
                #calculate cost and optimizer functions                    
                if(i!=(total_batch-1)):
                    gd_start = timeit.default_timer()

                    #train_args
                    #sess
                    #TRAIN_HERE
                    if(do_parallel_gd_and_mask==False or iepoch==0):
                        my_feed_dict={X: batch_x, Y: batch_y}
                        my_result = run_gd_step(sess, my_feed_dict, train_args)
                    else:
                        my_result = my_result_list[i]
                        if(verbose>0):
                            print("Result length:",len(my_result), flush=True)

                    if(save_summaries==True):
                        if(verbose>1):
                            summary, _, c, a, rl, sl, rh, g0, g1, myce0, myce1,mypt0, mypt1, mywce, myy = my_result
                        else:
                            summary, _, c, a, rl, myy = my_result
                        train_writer.add_run_metadata(run_metadata, 'k%03d-step%03d-batch%04d' % (ki, epoch, i) )
                        train_writer.add_summary(summary, epoch)
                    else:
                        if(verbose>1):
                            _, c, a, rl, sl, rh, g0, g1, myce0, myce1,mypt0, mypt1, mywce, myy = my_result
                        else:
                            if(calculate_r2_per_epoch==True):
                                _, c, a, rl, myy = my_result
                            else:
                                _, c, a, rl = my_result

                    gd_stop = timeit.default_timer()
                    gd_time += gd_stop-gd_start

                else:
                    gd_start = timeit.default_timer()

                    if(calculate_r2_per_epoch==True):     
                        if(save_summaries==True):
                            mysummary, a, c, myy, _, rl  = sess.run(pred_args, feed_dict={X: batch_x, Y: batch_y} )                
                            valid_writer.add_summary(mysummary, epoch )
                        else:
                            a, c, rl, myy = sess.run(pred_args, feed_dict={X: batch_x, Y: batch_y} )
                    else:
                        if(save_summaries==True):
                            mysummary, a, c, _, rl  = sess.run(pred_args, feed_dict={X: batch_x, Y: batch_y} )                
                            valid_writer.add_summary(mysummary, epoch )
                        else:
                            a, c, rl = sess.run(pred_args, feed_dict={X: batch_x, Y: batch_y} )
                            
                    gd_stop = timeit.default_timer()
                    gd_time += gd_stop-gd_start                        
            
                rest_start = timeit.default_timer()
            
                if(calculate_r2_per_epoch==True):
                    my_r2_tmp = pearson_r2(myy, batch_y)
                    avg_r2 += np.sum(my_r2_tmp['r2_per_sample'])/len(my_r2_tmp['r2_per_sample'])/total_batch

                if(calculate_acc_per_epoch==True):
                    avg_a += a/total_batch
                    
                avg_rl += rl/total_batch
                c_tmp=c/total_batch
                epoch_cost+=c_tmp
                avg_cost+=c_tmp/window
                if(verbose>1 and i==0):
                    print("batch cost c:",c, "Sparsity loss:", sl, "rho_hat range:", np.min(rh), np.max(rh))

                if(validate_after_epoch==True or (validate_after_cycle==True and cycle_count==repeat_cycles) ):
                    my_val_pred = vc, va, vrl, vmyy  = sess.run([cost, accuracy, reconstruction_loss, y_pred], feed_dict={X: batch_val_x, Y: batch_y} )    
                    my_vr2_tmp = pearson_r2(vmyy, batch_y)
                    val_avg_r2 += np.sum(my_vr2_tmp['r2_per_sample'])/len(my_vr2_tmp['r2_per_sample'])/total_batch
                    val_avg_a += va/total_batch
                    val_avg_rl += vrl/total_batch
                    val_epoch_cost+=vc/total_batch
                    val_avg_cost+=vc/total_batch/window

                if(save_activations==True and (iepoch==training_epochs or (epoch+1) % window == 0) ):
                    my_act = sess.run([encoder_operators[-1]], feed_dict={X: batch_x, Y: batch_y} )
                    act_dir = os.path.basename(sys.argv[1])+"_activations"
                    fname = act_dir + "/batch_" + str(i) + ".out"
                    print("Saving activations to", fname,flush=True)
                    with open(fname, 'w') as f:
                         np.savetxt(f,  my_act[0])
                    f.close()
                    print("Saving activations to complete.",flush=True)
                    if((i+1)==total_batch):
                        my_act = sess.run([encoder_operators[-1]], feed_dict={X: train_x[(i+1)*batch_size:], Y: train_y[(i+1)*batch_size:]} )
                        fname = act_dir + "/batch_" + str(i+1) + ".out"
                        print("Saving activations to", fname,flush=True)
                        with open(fname, 'w') as f:
                             np.savetxt(f,  my_act[0])
                        f.close()
                        print("Saving activations to complete.",flush=True)

                rest_stop = timeit.default_timer()
                rest_time += rest_stop-rest_start


            ############BATCH ENDS HERE

            rest_start = timeit.default_timer()

            if(save_pred==True): #if using cluster run for all k fold: if(save_pred==True):
                #This generates 1GB files per epoch per k-fold iteration, at least 5TB of space required, uncoment when using the HPC cluster with big storage
                my_pred = sess.run([y_pred], feed_dict={X: val_x, Y: val_y} )

                fname = "k-" + str(ki) + "_epoch-" + str(epoch+1) + "_val-pred.out"
                with open(fname, 'w') as f:
                    np.savetxt(f,  my_pred[0])
                f.close()

            if(epoch==early_stop_begin):
                #mask_increase
                mask_increase = 0.2*(repeat_cycles+1)/window
                #mask_rate
                mask_rate=0.8
            if(epoch>early_stop_begin and epoch % window == 0):
                #mask_rate
                mask_rate=0.8

                if(previous_window_avg_cost==0):                       
                    previous_window_avg_cost=avg_cost/(epoch/window)
                    current_window_avg_cost=avg_cost/(epoch/window)
                    print("previous_window_avg_cost: ", previous_window_avg_cost, " current_window_avg_cost: ", current_window_avg_cost)
                else:
                    current_window_avg_cost=avg_cost
                    print("previous_window_avg_cost: ", previous_window_avg_cost, " current_window_avg_cost: ", current_window_avg_cost)
                    if(current_window_avg_cost>(previous_window_avg_cost-(hysteresis*previous_window_avg_cost))):
                        time_to_stop=True
                    else:
                        previous_window_avg_cost=current_window_avg_cost
                avg_cost=0
                val_avg_cost=0

################Epoch end here
            if(verbose>1):
                print("Epoch ", epoch, " done. Masking rate used:", mask_rate, " Initial one:", initial_masking_rate, " Loss: ", epoch_cost, " Accuracy:", avg_a, " Reconstruction loss (" , loss_type, "): ", avg_rl, "Sparsity loss:", sl,"sr2:", avg_r2, flush=True)
                print("Shape pt0:", mypt0)
                print("Shape pt1:", mypt1)
                print("Shape g0:", g0)
                print("Shape g1:", g1)
                print("Shape ce0:", myce0)
                print("Shape ce1:", myce1)
                print("wce:", mywce)
                print("fl:", rl)
                print("myy:", myy)
            else:
                if(avg_r2 != 'DISABLED'):
                    avg_r2 = np.round(avg_r2, 6)
                if(avg_a != 'DISABLED'):
                    avg_a = np.round(avg_a, 6)
                avg_rl = np.round(avg_rl, 6)
                epoch_cost = np.round(epoch_cost, 6)
                print("Epoch ", epoch, " done. Masking rate used:", mask_rate, " Initial one:", initial_masking_rate, " Loss: ", epoch_cost, " Accuracy:", avg_a, " s_r2:", avg_r2, " Reconstruction loss (" , loss_type, "): ", avg_rl, flush=True)

            if(validate_after_epoch==True  or (validate_after_cycle==True and cycle_count==repeat_cycles) ):
                val_epoch_cost = np.round(val_epoch_cost,6)
                val_avg_a = np.round(val_avg_a, 6)
                val_avg_r2 = np.round(val_avg_r2, 6)
                val_avg_rl = np.round(val_avg_rl, 6)

                print("VALIDATION:Epoch ", epoch, " done. Masking rate used:", mask_rate, " Initial one:", initial_masking_rate, " Loss: ", val_epoch_cost, " Accuracy:", val_avg_a, " s_r2:", val_avg_r2, " Reconstruction loss (" , loss_type, "): ", val_avg_rl, flush=True)

            rest_stop = timeit.default_timer()
            rest_time += rest_stop-rest_start  
                            
            stop_epochs = timeit.default_timer()
            time_epochs += stop_epochs-start_epochs
            
            save_start = timeit.default_timer()

            if(save_model==True and (time_to_stop==True or iepoch==training_epochs or (epoch+1) % window == 0) ):
                #Create a saver object which will save all the variables
                saver = tf.train.Saver(max_to_keep=2)

                suffix = "best_M"+str(model_index)
                if(time_to_stop==True):
                    suffix = "last_M"+str(model_index)
                #Now, save the graph
                model_dir = os.path.basename(sys.argv[1])+"_model"
                filename = model_dir + "/inference_model-" + suffix + ".ckpt"
                print("Saving model to file:", filename)
                saver.save(sess, filename)
                print("Saving model done")

                if(custom_cmd!=''):
                    custom_result = run_custom_cmd()
                    print("CUSTOM:Epoch",epoch,"RESULT:", custom_result)

            save_stop = timeit.default_timer()
            save_time += save_stop-save_start

            if(time_to_stop==True):
                print("Stop criteria satisfied: epoch =", epoch, "previous_window_avg_cost =", previous_window_avg_cost, "current_window_avg_cost =", current_window_avg_cost)
                break

        if(save_pred==True):

            fname = "k-" + str(ki) + "_train-obs.out"
            with open(fname, 'w') as f:
                np.savetxt(f, train_y)
            f.close()

            fname = "k-" + str(ki) + "_train-input.out"
            with open(fname, 'w') as f:
                np.savetxt(f, train_x)
            f.close()

        if(detailed_metrics==True):
            print("Calculating summary statistics...")
        else:
            print("Skipping detailed statistics...")

        start_metrics = timeit.default_timer()

        if(save_summaries==True):
            mysummary, my_cost, my_rloss, my_sloss, my_acc, my_cacc  = sess.run([merged, cost, reconstruction_loss, sparsity_loss, accuracy, cost_accuracy], feed_dict={X: train_x, Y: train_y})

            valid_writer.add_summary(mysummary, epoch )
        
        if(full_train_report==False):
            my_cost = 0
            my_rloss = 0
            my_sloss = 0
            my_acc = 0
            my_cacc = 0
            my_F1 = [0,0,0]
            my_acc_t = [0,0,0,0]
            my_r2 = 0
            for i in range(total_batch):
                print("Calculating F1 for batch", i, flush=True)
                batch_x = train_x[i*batch_size:(i+1)*batch_size]
                batch_y = train_y[i*batch_size:(i+1)*batch_size]

                my_cost_tmp, my_rloss_tmp, my_sloss_tmp, my_acc_tmp, my_cacc_tmp, my_pred  = sess.run([cost, reconstruction_loss, sparsity_loss, accuracy, cost_accuracy, y_pred], feed_dict={X: batch_x, Y: batch_y})
                my_cost += my_cost_tmp/total_batch
                my_rloss += my_rloss_tmp/total_batch
                my_sloss += my_sloss_tmp/total_batch
                my_acc += my_acc_tmp/total_batch
                my_cacc += my_cacc_tmp/total_batch

                my_F1_tmp = f1_score(my_pred, batch_y, sess)
                my_r2_tmp = pearson_r2(my_pred, batch_y)
                my_r2 += np.sum(my_r2_tmp['r2_per_sample'])/len(my_r2_tmp['r2_per_sample'])/total_batch

                my_F1[0] += my_F1_tmp[0]/total_batch
                my_F1[1] += my_F1_tmp[1]/total_batch
                my_F1[2] += my_F1_tmp[2]/total_batch

                #RARE VS COMMON VARS
                print("Calculating accuracy per MAF threshold for batch", i)
                if(report_perf_by_rarity==True):
                    my_acc_tmp, _ = accuracy_maf_threshold_global(sess, my_pred, batch_y, rare_indexes)
                    my_acc_t[2] += my_acc_tmp/total_batch
                    my_acc_tmp, _ = accuracy_maf_threshold_global(sess, my_pred, batch_y, common_indexes)
                    my_acc_t[3] += my_acc_tmp/total_batch
                else:
                    my_acc_t[0], my_acc_t[1],my_acc_t[2],my_acc_t[3] = "NA","NA", "NA", "NA"


        else:
            my_cost, my_rloss, my_sloss, my_acc, my_cacc,my_pred  = sess.run([cost, reconstruction_loss, sparsity_loss, accuracy, cost_accuracy,y_pred], feed_dict={X: train_x, Y: train_y})
            my_F1 = f1_score(my_pred, train_y, sess)
            my_r2='NA'
            if(report_perf_by_rarity==True):
                my_acc_t[2], _ = accuracy_maf_threshold(sess, my_pred, train_y, MAF_all_var, rare_threshold1, rare_threshold2)
                my_acc_t[3], _ = accuracy_maf_threshold(sess, my_pred, train_y, MAF_all_var, common_threshold1, common_threshold2)
            else:
                my_acc_t[0], my_acc_t[1],my_acc_t[2],my_acc_t[3] = "NA","NA", "NA", "NA"

        r_report_start = timeit.default_timer()

        print("Accuracy [MAF",rare_threshold1,"-",rare_threshold2,"]", my_acc_t[2])
        print("Accuracy [MAF",common_threshold1,"-",common_threshold2,"]", my_acc_t[3])
        print("F1 score [MAF 0-1]:", my_F1)
        print("R-squared per sample:", my_r2)

        r_report_stop = timeit.default_timer()
        print("Time to calculate accuracy (rare versus common variants:", r_report_stop-r_report_start)
        r_report_time += r_report_stop-r_report_start

        if(detailed_metrics==True):

            my_pred = sess.run([y_pred], feed_dict={X: train_x, Y: train_y})
            my_pred = my_pred[0]
            print("Accuracy per veriant...")
            my_acc_t[0], acc_per_m = accuracy_maf_threshold(sess, my_pred, train_y, MAF_all_var, 0, 1)

            print("MSE per veriant...")
            my_MSE, MSE_list = MSE_maf_threshold(sess, my_pred, train_y, MAF_all_var, 0, 1)

            print("Emprirical r2hat per veriant...")
            mean_r2hat_emp, my_r2hat_emp = mean_empirical_r2(train_x, train_y)

        if(detailed_metrics==True):
            idx=1
            for i in range(len(MAF_all_var)):
                print("METRIC_MAF:", MAF_all_var[i])
                print("METRIC_acc_per_m:", acc_per_m[idx])
                print("METRIC_r2_emp:", my_r2hat_est[i])
                print("METRIC_r2_est:", my_r2hat_emp[i])
                print("METRIC_MSE_per_m:", MSE_list[idx])
                idx+=2


        mean_cost += my_cost
        mean_rloss += my_rloss
        mean_sloss += my_sloss
        mean_acc += my_acc
        mean_cacc += my_cacc
        mean_F1[0] += my_F1[0]
        mean_F1[1] += my_F1[1]
        mean_F1[2] += my_F1[2]

        if(detailed_metrics==True):
            for j in range(len(mean_acc_t)):
                mean_acc_t[j] += my_acc_t[j]

        stop_metrics = timeit.default_timer()

        time_metrics += stop_metrics-start_metrics

        print("Training complete.")
            
    stop = timeit.default_timer()
    
    print('Time to run all training (sec): ', stop - start)
    print('Time to run all epochs (sec): ', time_epochs)
    print('Time to run all gradient descent iteratons (GPU): ', gd_time)
    print('Time to run all masking iteratons (CPU): ', mask_time)
    print('Time to run all flatten iterations (CPU): ', flat_time)
    print('Time to run all shuffle iterations (CPU): ', shuf_time)
    print('Time to run all model saving (GPU<->CPU<->IO), once per checkpoint window: ', save_time)
    print('Time to define and start graph/session (CPU->GPU), ance by the start of training: ', prep_time)    
    print('Time to run all accuracy metrics (sec), once by the end of training: ', time_metrics)
#    print('Time to run all remaining tasks for all iterations (CPU): ', rest_time)

    print('Time to run each epoch, average (sec): ', time_epochs/(training_epochs+1))
    print('Time to run each gradient descent iteraton (GPU): ', gd_time/(training_epochs+1))
    print('Time to run each masking iteraton (CPU): ', mask_time/(training_epochs+1)) 
    print('Time to run each flatten iteration (CPU): ', flat_time/(training_epochs+1))
    print('Time to run each shuffle iteration (CPU): ', shuf_time/(training_epochs+1))
    print('Time to run performance per MAF threshold calculations (CPU<->GPU):', r_report_time)
    
    if(save_summaries==True):
        train_writer.close()
        valid_writer.close()   

    #reset tensorflow session
    tf.reset_default_graph()
    sess.close()
    # return the minimum loss of this combination of L1, L2, activation function, beta, rho on a dataset
    #return minLoss, min_reconstruction_loss, min_sparsity_loss, max_correl
    return mean_cost, my_rloss, mean_sloss, mean_acc, my_acc_t[2], my_acc_t[3], mean_F1[0], mean_F1[1], mean_F1[2], my_r2

 

def main():
    #split_size = 10 #k for 10-fold cross-validation
    
    print("Name of the script: ", sys.argv[0])
    print("Number of arguments: ", len(sys.argv)-1)
    print("The arguments are: " , str(sys.argv))

    if(save_model==True):
        model_dir=os.path.basename(sys.argv[1])+"_model"
        if(os.path.exists(model_dir)==False):
            os.mkdir(model_dir)

    if(save_activations==True):
        act_dir=os.path.basename(sys.argv[1])+"_activations"
        if(os.path.exists(act_dir)==False):
            os.mkdir(act_dir)

    #mask_rate=0.9
    
    global recovery_mode
    global initial_masking_rate
    global maximum_masking_rate
    global disable_masking
    global fixed_masking_rate   
        
    global gamma
    global loss_type
    global optimizer_type
    global hsize

    global left_buffer
    global right_buffer
    global model_path
    
    global mask_increase
    global keep_rate
    global NH
    global model_index

    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbose', type=int, default=0, dest='verbose', help="Increase output verbosity")
    parser.add_argument('-c', '--custom', type=str, default='', dest="custom_cmd", help="Custom validation script file to run after each checkpoint")
    parser.add_argument('argv', nargs='*')
    ar = parser.parse_args()

    global verbose
    verbose = ar.verbose
    global custom_cmd
    custom_cmd = ar.custom_cmd

    if(len(ar.argv)!=5 and len(ar.argv)!=6):
        return False

    print("Parsing input file: ")
        #Arguments
        #ar.argv[0] = [str] input file (HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf)
        #ar.argv[1] = [str] hyperparameter list file (mylist.txt)
        #ar.argv[2] = [True,False] Recovery mode, default is False
        #ar.argv[3] = [1/846] Initial masking rate
        #ar.argv[4] = [0.98] Final masking rate
        #ar.argv[5] = [str] only needed if recovery mode is True, path to the .ckpt file
    recovery_mode = ar.argv[2]
    #If enabling recovery mode to resume training, provide model path after masking rates, as last parameter
    if(recovery_mode=="True"):
        #path to the .ckpt file, example: ./recovery/inference_model-1.ckpt
        model_path=ar.argv[5]

    initial_masking_rate = convert_to_float(ar.argv[3])
    maximum_masking_rate = convert_to_float(ar.argv[4])

    if(maximum_masking_rate==0):
        disable_masking = True
    else:
        disable_masking = False

    if(maximum_masking_rate==initial_masking_rate):
        fixed_masking_rate = True
    else:
        fixed_masking_rate = False
        if(early_stop_begin==0):
            den=window
        else:
            den=early_stop_begin
        mask_increase = (maximum_masking_rate-initial_masking_rate)*(1+repeat_cycles)/den

    print("disable_masking =", disable_masking)
    print("fixed_masking_rate =", fixed_masking_rate)
    print("initial_masking_rate =", initial_masking_rate)
    print("maximum_masking_rate =", maximum_masking_rate)
    print("reading hyperparameter list from file: ", ar.argv[1])

    hp_array = []
    result_list = []

    with open(ar.argv[1]) as my_file:
        for line in my_file:
            if not line.startswith('#'):
                hp_array.append(line.split())

    i = 0
    print("number of hyperparameter sets found:", len(hp_array))

    while(i < len(hp_array)):

        l1 = float(hp_array[i][0]) #[float] L1 hyperparameter value
        l2 = float(hp_array[i][1]) #[float] L2 hyperparameter value
        beta = float(hp_array[i][2]) #[float] Sparsity beta hyperparameter value
        rho = float(hp_array[i][3]) #[float] Sparseness rho hyperparameter value
        act = str(hp_array[i][4]) #[str] Activation function ('tanh')
        lr = float(hp_array[i][5]) #[float] Learning rate hyperparameter value
        gamma = float(hp_array[i][6]) #[float] gamma hyper parameter value
        optimizer_type = str(hp_array[i][7]) #[string] optimizer type
        loss_type = str(hp_array[i][8]) #[string] loss function type
        hsize=str(hp_array[i][9]) #[float,string] hidden layer size
        left_buffer=int(hp_array[i][10]) #[int] left buffer size (number of features to be excluded from output layer start)
        right_buffer=int(hp_array[i][11]) #[int] right buffer size (number of features to be excluded from output layer end)
        keep_rate=convert_to_float(str(hp_array[i][12])) #[1,0.5] single float=keep prob of single hidden layer, float vector (0.5,1,1) = keep prob of each hidden layer, the number of hidden layers will be detected automatically from the number of keep_rate values provided, values are comma separated.
        if(',' in str(hp_array[i][12])):
            print("keep_rate:",keep_rate)
            NH=len(keep_rate)
        else:
            keep_rate=[keep_rate]
            print("keep_rate:",keep_rate)
            NH=1

        if(i==0):
            data_obs = process_data(ar.argv[0]) #input file, i.e: HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf

        tmp_list = hp_array[i]
        print("Starting autoencoder training. Model: ", str(model_index), "Parameters:", tmp_list)
        my_cost, my_rloss, my_sloss, my_acc,  my_racc, my_cacc, my_micro, my_macro, my_weighted, my_r2  = run_autoencoder(lr, training_epochs, l1, l2, act, beta, rho, data_obs)
        tmpResult = [my_cost, my_rloss, my_sloss, my_acc,  my_racc, my_cacc, my_micro, my_macro, my_weighted, my_r2]
        tmp_list.extend(tmpResult)
        result_list.append(tmp_list)
        i += 1
        model_index+=1
        print("TMP_RESULT: ", tmp_list)
    return result_list


if __name__ == "__main__":

    result = main()
    if(result==False):
        print("Please provide at least 5 arguments.\n",
            "Usage: script.py INVCF HP REC IM FM CKPT_DIR\n\n",
            "    INVCF = [string] input file in vcf format\n",
            "    HP = [string] hyperparameter list file (space delimited list of hyperparameters)\n",
            "    REC = [True,False] Recovery mode, default is False\n",
            "    IM = [float or fraction] Initial masking rate\n",
            "    FM = [float or fraction] Final masking rate\n",
            "    CKPT_DIR = [string] (optional) only needed if recovery mode is True, path to the .ckpt file\n\n",
            "Example 1 (new training): script.py my_input.vcf my_param.txt False 0.1 0.99\n",
            "Example 2 (resume previous training): script.py my_input.vcf my_param.txt True 0.1 0.99 ./my_dir/my_model.ckpt\n\n",
            "Where my_param.txt (hyperparameter file) should contain the following hyperparameters:,\n\n",
            "    L1: [float] L1 (Lasso) regularizer, small values recommended, should be less than 1, typically between 1e-2 and 1e-8\n",
            "    L2: [float] L2 (Ridge) regularizer, small values recommended, should be less than 1, typically between 1e-2 and 1e-8\n",
            "    beta: [float] Sparsity scaling factor beta, any value grater than 1\n",
            "    rho: [float] Desired average hidden layer activation (rho), less than 1\n",
            "    act: [string] Activation function type, values supported: ['sigmoid','tanh', 'relu']\n",
            "    LR: [float] Learning rate\n",
            "    gamma: [float] scaling factor for focal loss, ignored when loss_type!=FL\n",
            "    optimizer: [string] optimizer type, values supported: ['GradientDescent', 'Adam', 'RMSProp']\n",
            "    loss_type: [string] loss type, values supported: ['MSE', 'CE', 'WCE', 'FL'], which respectively mean: mean squared error, cross entropy, weighted cross entropy, and focal loss.\n",
            "    h_size: [float,string] hidden layer size, if 'sqrt' the hidden layer size will be equal to the square root of the input layer size, if float, the hidden layer size will be the hyperparameter value multiplied by input layer size\n",
            "    LB: [int] left buffer size, number of upstream input variants to be excluded from output layer\n",
            "    RB: [int] right buffer size, number of downstream input variants to be excluded from output layer\n",
            "    KP: [float vector] keep probability of each hidden layer, the number of hidden layers will be detected automatically from the number of values provided, each value must be comma separated (e.g: 1.0,0.8,0.3 for 3 hidden layers; 1.0,0.5 for 2 hidden layers, 0.5 for 1 hidden layer, etc.)\n\n",
            "Each hyperparamter should be separated by space or tab, one hyper parameter set per line, for example (cat my_param.txt):\n\n",
            "    #L1, L2, BETA, RHO, ACT, LR, gamma, optimizer, loss_type, h_size, LB, RB, KP\n",
            "    1e-06 1e-06 0.001 0.07 relu 10 5 RMSProp WCE sqrt 0 12 1\n",
            "    1e-05 1e-08 6 0.04 tanh 1e-05 2 GradientDescent FL 1 23 11 1,0.5\n",
            "    0.01 0.0001 0.01 0.004 tanh 0.0001 0 Adam FL 0.5 10 0 1,1,0.3\n")
        sys.exit()
        
    print("LABELS  [L1, L2, BETA, RHO, ACT, LR, gamma, optimizer, loss_type, h_size, LB, RB, rsloss, rloss, sloss, acc, ac_r, ac_c, F1_micro, F1_macro, F1_weighted, sr2]") 
    i = 0
    while(i < len(result)):
        print("RESULT ", result[i])
        i += 1
