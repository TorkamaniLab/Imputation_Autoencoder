# coding: utf-8

#current update: enable parallel gradient descent step and masking
#previous update: enabled support for cross entropy loss (CE), weighted CE, and focal loss, suuport for multiple optimizers

#Batch mode example
#python3.6 ../10-fold_CV_imputation_autoencoder_from_grid_search_v3_online_data_augmentation_parallel.py ../HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf.clean3.subset1000 best_hp_set.txt False 0 0
#/bin/python3.6 script_name.py imputed_test_subset.vcf 3_hyper_par_set.txt False 0 0
#CUDA_VISIBLE_DEVICES=0 /bin/python3.6 ../script_name.py imputed_test_subset.vcf 3_hyper_par_set.txt True 0 0

#sequential mode
#python3.6 ../10fold_CV_imputation_autoencoder_from_grid_search_v3_online_data_augmentation_parallel.py ../HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf.clean4 0.039497493748 0.001096668917 0.001 0.021708661247 sigmoid 5.6489904e-05 3 Adam FL False 0 0

import math #sqrt
import tensorflow as tf
import numpy as np #pip install numpy==1.16.4
import pandas as pd

import random #masking
# sorting results
from collections import defaultdict
from operator import itemgetter

import timeit #measure runtime

#from tqdm import tqdm # progress bar

from scipy.stats import pearsonr #remove this, nan bugs are not corrected

from tensorflow.python.client import device_lib #get_available_devices()

from scipy.stats import linregress
#linregress(a, b)[2] gives the correlation, correcting for nans due to monomorphic predictions
#linregress(a, b)[3] gives the p-value correl 
#a are all the predicted genotypes for one given SNP, in ALT dosage form, b is the same but for experimental SNPs

import sys #arguments
import operator #remove entire columns from 2d arrays

import subprocess as sp #run bash commands that are much faster than in python (i.e cut, grep, awk, etc)

import os

#parallel processing libraries
import multiprocessing
from functools import partial # pool.map with multiple args


import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#try:
#    from tensorflow.python.util import module_wrapper as deprecation
#except ImportError:
#    from tensorflow.python.util import deprecation_wrapper as deprecation
#deprecation._PER_MODULE_WARNING_LIMIT = 0


###################################DEV_OPTIONS#############################################

############Performance options: data loading
do_parallel = False #load data and preprocess it in parallel
do_parallel_MAF = False #also do MAF calculation in parallel, otherwise use PLINK

############Performance options: masking
do_numpy_masking = True #array based method, much faster then nested for loops!
par_mask_proc = 20 #how many masking parallel threads or processes, only used if par_mask_method is not "serial"
par_mask_method = "threadpool" #["serial","threadpool","thread","joblib","ray"] "serial" to disable, parallel masking method, threadpool has been the fastest I found
do_parallel_numpy_per_cycle = True #whole cycle in parallel
do_parallel_gd_and_mask = False #instead of parallelizing masking only, run masking and gradient descent in parallel

############Performance options: future improvements
use_cuDF = False #TODO enable data loading directly in the GPU

############backup options and reporting options
save_model = True #[True,False] save final model generated after all training epochs, generating a checkpoint that can be loaded and continued later
save_pred = False #[True,False] save predicted results for each training epoch and for k-fold CV
resuming_step = 3001 #number of first step if recovery mode True
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
categorical = "False" #False: treat variables as numeric allele count vectors [0,2], True: treat variables as categorical values (0,1,2)(Ref, Het., Alt)
split_size = 100 #number of batches
training_epochs = 499 #learning epochs (if fixed masking = True) or learning permutations (if fixed_masking = False), default 25000, number of epochs or data augmentation permutations (in data augmentation mode when fixed_masking = False)
#761 permutations will start masking 1 marker at a time, and will finish masking 90% of markers
last_batch_validation = False #when k==1, you may use the last batch for valitation if you want
alt_signal_only = False #TODO Wether to treat variables as alternative allele signal only, like Minimac4, estimating the alt dosage
all_sparse=True #set all hidden layers as sparse
custom_beta=False #if True, beta scaling factor is proportional to number of features
average_loss=False #True/False use everage loss, otherwise total sum will be calculated
disable_alpha=True #disable alpha for debugging only
inverse_alpha=False
early_stop_begin=1 #after what epoch to start monitoring the early stop criteria
window=500 #stop criteria, threshold on how many epochs without improvement in average loss, if no improvent is observed, then interrupt training
hysteresis=0.0001 #stop criteria, improvement ratio, extra room in the threshold of loss value to detect improvement, used to identify the beggining of a performance plateau

############Masking options
fixed_masking = False #True: mask variants only at the beggining of the training cycle, False: mask again with a different pattern after each iteration (data augmentation mode)
mask_per_sample = True #True: randomly mask genotypes per sample instead of mask the entire variant for all samples, False: mask the entire variant for all samples 
random_masking = True #set random masks instead of preset ones
mask_preset = False #True: mask from genotype array
shuffle = True #Whether shuffle data or not at the begining of training cycle. Not necessary for online data augmentation.
repeat_cycles = 4 #how many times to repeat the masking rate
validate_after_epoch=False #after each epoch, apply a new masking pattern, then calculate validation accuracy on the new unseen mask pattern
calculate_r2_per_epoch=False
calculate_acc_per_epoch=False

############debugging options
verbose=0

###################################OPTIONS#############################################
 
#global variables
MAF_all_var = [] #MAF calculated only once for all variants,remove redundancies
rare_indexes = [] #indexes of rare variants
common_indexes = [] #indexes of common variants
MAF_all_var_vector = [] #vector of weights equal to number of output nodes (featuresX2 by default)
ncores = multiprocessing.cpu_count() #for parallel processing
config = tf.ConfigProto(log_device_placement=False)
config.intra_op_parallelism_threads = 0 #0=auto
config.inter_op_parallelism_threads = 0 #0=auto
config.gpu_options.allow_growth=True
model_index=0

if(par_mask_method == "joblib"):
    import joblib
elif(par_mask_method == "thread"):
    import threading, queue
elif(par_mask_method == "ray"):
    import ray

def convert_gt_to_int(gt,alt_signal_only=False):

    genotype_to_int={'0/0': [1,0], '0|0': [1,0], '0/1': [1,1], '0|1':[1,1], '1/0':[1,1], '1|0':[1,1], '1/1':[0,1], '1|1':[0,1], './0':[0,0], './1':[0,0], './.':[0,0], '0/.':[0,0], '1/.':[0,0]}
    result=genotype_to_int[gt[0:3]]

    if(alt_signal_only==True):
        genotype_to_int={'0/0': 0, '0|0': 0.0, '0/1': 1, '0|1':1, '1/0':1, '1|0':1, '1/1':2, '1|1':2, './0':-1, './1':-1, './.':-1, '0/.':-1, '1/.':-1}
    
    return result


def process_lines(lines):

    result_line=[]
    pos=0
    chr=0
    start_pos='0'
    first=True

    snp_ids={}

    processed_result=[]
    
    for line in lines:
        #skip comments
        if(line[0]=='#'):
            continue
        if(first==True):
            chr, start_pos = line.split('\t')[0:2]
            first=False

        vcf_line=line.split('\t')
        vcf_line[-1]=vcf_line[-1].replace('\n','')
        for column in vcf_line[9:]:
            result=convert_gt_to_int(column[0:3])
            result_line.append(result)
            
        processed_result.append(result_line)
        
    return processed_result

def convert_genotypes_to_int(indexes, file, categorical="False"):
    if(verbose>0):
        print("process:", multiprocessing.current_process().name, "arguments:", indexes, ":", file)
    
    j=0
    #command = "cut -f"
    #for i in range(len(indexes)):
    #    command = command + str(indexes[i]+1)
    #    if(i<len(indexes)-1):
    #        command = command + ","

    #command = command + " " + file
    #print(command)

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
    
    #print("BATCH SHAPE: ", len(df), len(df[0]))
    #print(df[0])
    new_df = 0
    if(categorical=="False"):
        new_df = np.zeros((len(df),len(df[0]),2))
        #new_df = np.zeros((df.shape[1]-9,len(df)*2))
    else:
        new_df = np.zeros((len(df),len(df[0])))
    #print(type(df))
    i = 0 #RR column index
    j = 0 #RR row index
    idx = 0
    my_hom = 2
    
    if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
        my_hom = 1
    
    #print(len(df), df[0][0])
    #print(len(df[0]))
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
        #pbar.update(1)
        idx += 1

    #print("processed_data")
    #for i in range(10):
    #    print(new_df[i][0])

    #the data needs to be flattened because the matrix multiplication step (x*W) 
    #doesn't support features with subfeatures (matrix of vectors)
    #new_df = np.reshape(new_df, (new_df.shape[0],new_df.shape[1]*2))

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
    current_train_y = flatten_data(my_sess, train_y)
    current_train_x = flatten_data(my_sess, current_train_x)
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

def submit_mask_tasks(my_data, mask_rate, categorical):
    
    par_tasks = 1
    if(do_parallel_numpy_per_cycle==True):
        par_tasks += repeat_cycles
    
    pool = multiprocessing.pool.ThreadPool(par_tasks)
    #pool = multiprocessing.Pool(par_tasks)

    result = []
    for i in range(par_tasks):
        result.append(pool.apply_async(partial(mask_data_per_sample_parallel, mydata=np.copy(my_data), mask_rate=mask_rate, categorical=categorical),[i]))
    
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

def run_parallel_gd_and_mask(sess, X, Y, current_train_x, my_data, mask_rate, categorical, my_args, batch_size):
   
    m_result, pool = submit_mask_tasks(my_data, mask_rate, categorical)
        
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


def process_data(file, categorical="False"):
     
    #Header and column names start with hashtag, skip those
    ncols = pd.read_csv(file, sep='\t', comment='#',header=None, nrows=1)    
    ncols = ncols.shape[1]
    
    print("Processing input data.")
    print("categorical: ", categorical)
    n_samples = ncols-9
    #RR subtract 9 from the number of columns to get the number of SNPs, 
    #RR because the first 9 columns are variant information, not genotypes
    print("number of samples: ", n_samples)
    
    indexes = list(range(10,ncols+1)) #bash cut index is 1-based
            
    start = timeit.default_timer()
    
    if(do_parallel==False):
        results = convert_genotypes_to_int(indexes, file, categorical)
        print( len(results), len(results[0]), len(results[0][0]))

    else:
        chunks = chunk(indexes, ncores )        

        pool = multiprocessing.Pool(ncores)

        results = pool.map(partial(convert_genotypes_to_int, file=file, categorical=categorical),chunks)
      
        pool.close()
        pool.join()
                        
        print(len(results), len(results[0]), len(results[0][0]) , len(results[0][0][0]))
    
        #for i in range(len(results)):
        #    print(len(results[i]))
    
        #merge outputs from all processes, reshaping nested list
        results = [item for sublist in results for item in sublist]

    print(len(results), len(results[0]), len(results[0][0]) )

    print("This file contains {} features (SNPs) and {} samples (subjects)".format(len(results[0]), n_samples))
    
    indexes = list(range(len(results[0])))

    results = np.asarray(results)
    
    stop = timeit.default_timer()
    print('Time to load the data (sec): ', stop - start)
    
    start_time = timeit.default_timer()

    global MAF_all_var
    
    if(do_parallel_MAF == False):

        #MAF_all_var = calculate_MAF_global(indexes, results, categorical)
        MAF_all_var = calculate_ref_MAF(file)

    else:
        chunks = chunk(indexes,ncores)

        pool = multiprocessing.Pool(ncores)

        MAF_all_var = pool.map(partial(calculate_MAF_global, inx=results, categorical=categorical),chunks)

        pool.close()
        pool.join()

        #merge outputs from all processes, reshaping nested list
        MAF_all_var = [item for sublist in MAF_all_var for item in sublist]

    global MAF_all_var_vector
    MAF_all_var_vector = []

    #create list of variants that belong to output layer, 1 index per variant
    keep_indexes=list(range(left_buffer,len(MAF_all_var)-right_buffer))
    
    #find out the layer structure (either 2 or 3 features per variant)
    n=2
    if(categorical==True):
        n=3

    for i in keep_indexes:
        MAF_all_var_vector.append(MAF_all_var[i])
        MAF_all_var_vector.append(MAF_all_var[i])
        if(categorical==True):
            MAF_all_var_vector.append(MAF_all_var[i])

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
    rare_indexes = filter_by_MAF_global(results, MAF_all_var, threshold1=rare_threshold1, threshold2=rare_threshold2, categorical=categorical)    
    common_indexes = filter_by_MAF_global(results, MAF_all_var, threshold1=common_threshold1, threshold2=common_threshold2, categorical=categorical)
    
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
    print("LENGTH2", len(MAF_all_var_vector)) 
    stop_time = timeit.default_timer()
    print('Time to calculate MAF (sec): ', stop_time - start_time)

    return results

def filter_by_MAF_global(x, MAFs, threshold1=0, threshold2=1, categorical=False):
    
    #don't do any filtering if the thresholds are 0 and 1
    if(threshold1==0 and threshold2==1):
        return x

    indexes_to_keep = []
    i = 0
    j = 0
    k = 0   
    
    while i < len(MAFs):
        if(MAFs[i]>threshold1 and MAFs[i]<=threshold2):
            if(categorical==True or categorical=="True"):
                indexes_to_keep.append(j)
                indexes_to_keep.append(j+1)
                indexes_to_keep.append(j+2)
            elif(categorical==False or categorical=="False"):
                indexes_to_keep.append(k)
                indexes_to_keep.append(k+1)            
        i += 1
        j += 3
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
    result = sp.check_output("/gpfs/home/raqueld/bin/plink --vcf "+refname+" --freq --out "+outname, encoding='UTF-8', shell=True)

    MAF_all_var = read_MAF_file(outname+".frq")

    return MAF_all_var

def mask_data_per_sample_parallel(i,mydata, mask_rate=0.9, categorical="False"):

    if(verbose>0):
        print("Data to mask shape:", mydata.shape, flush=False)
        
    nmask = int(round(len(mydata[0])*mask_rate))
    my_mask=[0,0]

    if(categorical=="True"):
        my_mask=-1
    elif(alt_signal_only==True):
        my_mask=0

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
            #a little slower if you restart ray every time
            #ray.shutdown()
            #ray.init()
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
            #ray.shutdown()
            
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

def logfunc(x, x2):
    x = tf.cast(x, tf.float64)
    x2 = tf.cast(x2, tf.float64)
    
    eps=tf.cast(1e-14, tf.float64)
    one=tf.cast(1.0, tf.float64)
    eps2 = tf.subtract(one,eps)
    
    cx = tf.clip_by_value(x, eps, eps2)
    cx2 = tf.clip_by_value(x2, eps, eps2)
    return tf.multiply( x, tf.log(tf.div(cx,cx2)))


#Kullback-Leibler divergence equation (KL divergence)
#The result of this equation is added to the loss function result as an additional penalty to the loss based on sparsity
def KL_Div(rho, rho_hat):

    rho = tf.cast(rho, tf.float64)
    rho_hat = tf.cast(rho_hat, tf.float64)

    KL_loss = rho * logfunc(rho, rho_hat) + (1 - rho) * logfunc((1 - rho), (1 - rho_hat))
    
    #rescaling KL result to 0-1 range
    return tf.div(KL_loss-tf.reduce_min(KL_loss)+1e-10,tf.reduce_max(KL_loss)-tf.reduce_min(KL_loss))

    #RR I just simplified the KL divergence equation according to the book:
    #RR "Hands-On Machine Learning with Scikit-Learn and TensorFlow" by Aurélien Géron
    #RR Example source code here https://github.com/zhiweiuu/sparse-autoencoder-tensorflow/blob/master/SparseAutoEncoder.py
    #RR KL2 is the classic sparsity implementation, source reference: https://github.com/elykcoldster/sparse_autoencoder/blob/master/mnist_sae.py
    #https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
def pearson_r_loss(y_true, y_pred):
    
    #y_true = tf.cast(y_true, tf.float32)
    #y_pred = tf.cast(y_pred, tf.float32)
    
    pearson_r, update_op = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true, name='pearson_r')
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'pearson_r'  in i.name.split('/')]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        pearson_r = tf.identity(pearson_r)
        pearson_r = tf.square(pearson_r)
        rloss = tf.subtract(1.0, pearson_r, name='reconstruction_loss')
        #return 1-pearson_r**2
        return rloss

def weighted_MSE(y_pred, y_true):
    MSE_per_var = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred)), axis=0)
    
    #The condition tensor acts as a mask that chooses, based on the value at each element, whether the corresponding element / row in the output should be taken from x (if true) or y (if false).
    #cond_shape = y_true.get_shape()
    #x = tf.ones(cond_shape)
    #y = tf.zeros(cond_shape)
    
    #tf.where(tf.equal(0,tf.round(y_true)), x, y)       
    #mean(((1.5-MAF)^5)*MSE)
    weights = tf.subtract(1.5, MAF_all_var_vector)
    weights = tf.pow(weights, gamma)
    weighted_MSE_per_var = tf.multiply(MSE_per_var, weights)
     
    weighted_MSE_loss = tf.reduce_mean(weighted_MSE_per_var, name='reconstruction_loss')
    return weighted_MSE_loss


def calculate_pt(y_pred, y_true):

    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.cast(y_true,tf.float64)

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

    #ref:https://github.com/unsky/focal-loss/blob/master/focal_loss.py
    pt_1 = tf.clip_by_value(pt_1, 1e-10, 1.0-1e-10) #avoid log(0) that returns inf
    #pt_1 = tf.add(pt_1, 1e-8) #avoid log(0) that returns inf

    #if value is zero in y_true, than take value from y_pred, otherwise, write zeros
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_0 = tf.clip_by_value(pt_0, 1e-10, 1.0-1e-10)
        
    return pt_0, pt_1

def calculate_CE(y_pred, y_true):

    pt_0, pt_1 = calculate_pt(y_pred, y_true)
    one = tf.cast(1.0, tf.float64)
    #eps=tf.cast(1.0+1e-8, tf.float64)
    n1 =  tf.cast(-1.0, tf.float64)

    CE_1 = tf.multiply(n1,tf.log(pt_1))
    CE_0 = tf.multiply(n1,tf.log(tf.subtract(one,pt_0)))

    return CE_0, CE_1

def cross_entropy(y_pred, y_true):

    pt_0, pt_1 = calculate_pt(y_pred, y_true)

    CE_0, CE_1 =  calculate_CE(y_pred, y_true)

    CE_1 = tf.reduce_sum(CE_1)
    CE_0 = tf.reduce_sum(CE_0)

    if(average_loss==True):
        CE = tf.divide(tf.add(CE_1, CE_0), len(y_true)*len(y_true[0]), name='reconstruction_loss')
    else:
        CE = tf.add(CE_1, CE_0, name='reconstruction_loss')

    return CE

def calculate_alpha():

    one=tf.cast(1.0, tf.float64)
    eps=tf.cast(1.0-1e-4, tf.float64)

    alpha = tf.multiply(tf.cast(MAF_all_var_vector,tf.float64),2.0)
    alpha = tf.clip_by_value(alpha, 1e-4, eps)

    if(inverse_alpha==True):
        alpha_1 = tf.divide(one, alpha)
        alpha_0 = tf.divide(one, tf.subtract(one,alpha))
    else:
        alpha_1 = alpha
        alpha_0 = tf.subtract(one,alpha)
        
    return alpha_0, alpha_1

def weighted_cross_entropy(y_pred, y_true):

    one=tf.cast(1.0, tf.float64)
    eps=tf.cast(1.0+1e-8, tf.float64)
    n1 =  tf.cast(-1.0, tf.float64)

    #pt_0, pt_1 = calculate_pt(y_pred, y_true)

    CE_0, CE_1 =  calculate_CE(y_pred, y_true)

    alpha_0, alpha_1 = calculate_alpha()

    WCE_per_var_1 = tf.multiply(CE_1, alpha_1)
    WCE_per_var_0 = tf.multiply(CE_0, alpha_0)

    WCE_1 = tf.reduce_sum(WCE_per_var_1)
    WCE_0 = tf.reduce_sum(WCE_per_var_0)

    if(average_loss==True):
        WCE = tf.divide(tf.add(WCE_1, WCE_0), len(y_true)*len(y_true[0]), name='reconstruction_loss')
    else:
        WCE = tf.add(WCE_1, WCE_0, name='reconstruction_loss')

    return WCE


def calculate_gamma(y_pred, y_true):
    
    one=tf.cast(1.0, tf.float64)
    #eps=tf.cast(1.0+1e-10, tf.float64)

    my_gamma=tf.cast(gamma, tf.float64)

    pt_0, pt_1 = calculate_pt(y_pred, y_true)

    #if statement to avoid useless calculaions
    if(gamma == 0):
        gamma_0 = one
        gamma_1 = one
    elif(gamma == 1):
        gamma_0 = pt_0
        gamma_1 = tf.subtract(one, pt_1)
    elif(gamma == 0.5):
        gamma_0 = tf.sqrt(pt_0)
        gamma_1 = tf.sqrt(tf.subtract(one, pt_1))
    else:
        gamma_0 = tf.pow(pt_0, my_gamma)
        gamma_1 = tf.pow(tf.subtract(one, pt_1), my_gamma)
    
    return gamma_0, gamma_1

#ref: https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
def focal_loss(y_pred, y_true):
    
    one=tf.cast(1.0, tf.float64)
    
    CE_0, CE_1 =  calculate_CE(y_pred, y_true)

    alpha_0, alpha_1 = calculate_alpha()

    FL_per_var_1_a = CE_1[:,1::2]
    FL_per_var_0_a = CE_0[:,1::2]
    FL_per_var_1_r = CE_1[:,0::2]
    FL_per_var_0_r = CE_0[:,0::2]
    #avoid useless calculations
    if(gamma>0):
        gamma_0, gamma_1 = calculate_gamma(y_pred[:,1::2], y_true[:,1::2])

        FL_per_var_1_a = tf.multiply(gamma_1, FL_per_var_1_a)
        FL_per_var_0_a = tf.multiply(gamma_0, FL_per_var_0_a)
        
        gamma_0, gamma_1 = calculate_gamma(y_pred[:,0::2], y_true[:,0::2])

        FL_per_var_1_r = tf.multiply(gamma_1, FL_per_var_1_r)
        FL_per_var_0_r = tf.multiply(gamma_0, FL_per_var_0_r)
     
    if(disable_alpha==False):
        #extract and reweight alternative allele
        FL_per_var_1_a = tf.multiply(FL_per_var_1_a, alpha_1[1::2])
        FL_per_var_0_a = tf.multiply(FL_per_var_0_a, alpha_0[1::2])
        #extract and reweight reference allele
        FL_per_var_1_r = tf.multiply(FL_per_var_1_r, alpha_0[0::2])
        FL_per_var_0_r = tf.multiply(FL_per_var_0_r, alpha_1[0::2])
 
    FL_1_a = tf.reduce_sum(FL_per_var_1_a)
    FL_0_a = tf.reduce_sum(FL_per_var_0_a)
    FL_1_r = tf.reduce_sum(FL_per_var_1_r)
    FL_0_r = tf.reduce_sum(FL_per_var_0_r)

    FL_1 = tf.add(FL_1_a, FL_0_a)
    FL_0 = tf.add(FL_1_r, FL_0_r)

    if(average_loss==True):
        FL = tf.divide(tf.add(FL_1, FL_0), tf.multiply(y_pred.get_shape()[0],y_pred.get_shape()[1]) , name='reconstruction_loss')
    else:
        FL = tf.add(FL_1, FL_0, name='reconstruction_loss')

    return FL

def fl01(y_pred, y_true):
    
    #avoid making useless calculations if gamma==0
    #if(gamma==0):
    #    WCE = weighted_cross_entropy(y_pred, y_true)
    #    return WCE

    one=tf.cast(1.0, tf.float64)
    #eps=tf.cast(1.0+1e-8, tf.float64)
    
    CE_0, CE_1 =  calculate_CE(y_pred, y_true)

    alpha_0, alpha_1 = calculate_alpha()

    FL_per_var_1 = tf.multiply(CE_1, alpha_1)
    FL_per_var_0 = tf.multiply(CE_0, alpha_0)
     
    FL_per_var_1 = tf.multiply(FL_per_var_1, gamma_1)
    FL_per_var_0 = tf.multiply(FL_per_var_0, gamma_0)

    FL_1 = tf.reduce_sum(FL_per_var_1)
    FL_0 = tf.reduce_sum(FL_per_var_0)

    return FL_0, FL_1

def f1_score(y_pred, y_true, sess):
    
    f1s = [0, 0, 0]
    two = tf.cast(2.0, tf.float64)
    eps = tf.cast(1e-8, tf.float64)

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    
    y_true = tf.clip_by_value(y_true, 0.0, 1.0) #in case the input encoding is [0,1,2]
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(tf.multiply(y_pred, y_true), axis=axis)
        FP = tf.count_nonzero(tf.multiply(y_pred, tf.subtract(y_true,1.0)), axis=axis)
        FN = tf.count_nonzero(tf.multiply(tf.subtract(y_pred,1.0),y_true), axis=axis)
        
        TP = tf.cast(TP, tf.float64)
        FP = tf.cast(FP, tf.float64)
        FN = tf.cast(FN, tf.float64)
        
        TP = tf.add(TP, eps)
        
        precision = tf.divide(TP, tf.add(TP, FP))
        recall = tf.divide(TP, tf.add(TP, FN))
        #f1 = tf.multiply(two, tf.multiply(precision, tf.divide(recall, tf.add(precision, recall))))
        top = tf.multiply(precision, recall)
        bottom = tf.add(precision, recall)
        f1 = tf.multiply(two, tf.divide(top,bottom))

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights = tf.divide(weights,tf.reduce_sum(weights))

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

    eps=1e-15

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
    x=tf.cast(x, tf.float64)
    
    print("Setting up encoder/decoder.")
    if(l2_val==0):
        regularizer = tf.contrib.layers.l1_regularizer(l1_val)
    else:
        regularizer = tf.contrib.layers.l1_l2_regularizer(l1_val,l2_val)

    print("keep_rate:", keep_rate)
    #dropout
    if keep_rate != 1: ##RR added dropout
            x = dropout(x, 'x', keep_rate) ##RR added dropout
              
    if func == 'sigmoid':
        print('Encoder Activation function: sigmoid')
        
        #tf.nn.sigmoid computes sigmoid of x element-wise.
        #Specifically, y = 1 / (1 + exp(-x))
        #tf.matmul multiply output of input_layer with a weight matrix and add biases
        #tf.matmul Multiplies matrix a by matrix b, producing a * b
        #If one or both of the matrices contain a lot of zeros, a more efficient multiplication algorithm can be used by setting the corresponding a_is_sparse or b_is_sparse flag to True. These are False by default.
        #tf.add will sum the result from tf.matmul (input*weights) to the biases
        #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        #with tf.device("/gpu:1"):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

        #This layer implements the operation: 
        #outputs = activation(inputs * weights + bias) 
        #where activation is the activation function passed as the activation argument, defined inprevious line
        #The function is applied after the linear transformation of inputs, not W*activation(inputs)
        #Otherwise the function would not allow nonlinearities
        layer_1 = tf.layers.dense(layer_1, units=units_num, kernel_regularizer= regularizer)
                       
    elif func == 'tanh':
        print('Encoder Activation function: tanh')
        #with tf.device("/gpu:1"):
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_1 = tf.layers.dense(layer_1, units=units_num, kernel_regularizer= regularizer)
        #layer_1 = tf.layers.dense(layer_1, units=221, kernel_regularizer= regularizer)
    
    elif(func == 'relu' or func == 'relu,sigmoid' or func == 'relu,tanh' or func == 'relu_sigmoid' or func == 'relu_tanh'):
        print('Encoder Activation function: relu')
        #with tf.device("/gpu:1"):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
        layer_1 = tf.layers.dense(layer_1, units=units_num, kernel_regularizer= regularizer)
        #layer_1 = tf.layers.dense(layer_1, units=221, kernel_regularizer= regularizer)

    return layer_1
        
def decoder(x, func, weights, biases):
    
    x = tf.cast(x, tf.float64)
    
    if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
        entropy_loss = True
    else:
        entropy_loss = False
        
    if(func == 'sigmoid' or func == 'relu,sigmoid' or func == 'relu_sigmoid'):
        print('Decoder Activation function: sigmoid')
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        #rescaling, if dealing with categorical variables or factors, tf.reduce_max(x) will result in 1
        if(entropy_loss==False):
            layer_1 = tf.multiply(layer_1, tf.reduce_max(x), name="y_pred")
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    elif(func == 'tanh' or func == 'relu,tanh' or func == 'relu_tanh'):
        print('Decoder Activation function: tanh')
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        #rescaling, if dealing with categorical variables or factors, tf.reduce_max(x) will result in 1
        if(entropy_loss==False):
            layer_1 = tf.div(tf.multiply(tf.add(layer_1, 1), tf.reduce_max(x)), 2, name="y_pred")
        else:
            layer_1 = tf.div(tf.add(layer_1, 1), 2, name="y_pred")
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    elif func == 'relu':
        print('Decoder Activation function: relu')
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']), name="y_pred")
        #no rescaling needed
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    return layer_1


def mean_empirical_r2(x,y, categorical=False):
    #This calculates exactly the same r2 as the empirical r2hat from minimac3 and minimac4
    #The estimated r2hat is different
    j=0
    mean_correl = 0
    correls = []
    while j < len(y[0]):
        if(categorical==False):
            getter = operator.itemgetter([j+1])
            x_genotypes = list(map(list, map(getter, np.copy(x))))
            y_genotypes = list(map(list, map(getter, np.copy(y))))
            x_genotypes = list(np.array(x_genotypes).flat)
            y_genotypes = list(np.array(y_genotypes).flat)
            #print("GGGGGG")
            #print(x_genotypes)
            #print(y_genotypes)
            correl = linregress(x_genotypes, y_genotypes)[2]
            mean_correl += correl/len(y[0])
            j+=2
        else:
            x_genotypes = []
            y_genotypes = []
            for i in range(len(y)):
                x_genotypes.append(np.argmax(x[i][j:j+3]))                
                y_genotypes.append(np.argmax(y[i][j:j+3]))
            
            correl = linregress(x_genotypes, y_genotypes)[2]
            mean_correl += correl/len(y[0])
            j+=3
        correls.append(mean_correl)
        #print("mean_correl",mean_correl)
    return mean_correl, correls

def calculate_MAF(x, categorical=False):
    j=0
    MAF_list = []
    if(categorical==True):
        while j < (len(x[0])-2):
            ref = 0
            alt = 0
            MAF = 0        
            for i in range(len(x)):
                allele_index = np.argmax(x[i][j:j+3])
                if(allele_index == 0):
                    ref+=2
                elif(allele_index == 1):
                    ref+=1
                    alt+=1
                elif(allele_index == 2):
                    alt+=2
            if(alt<=ref):
                MAF=alt/(ref+alt)
                #major=ref/len(y)
            else:
                MAF=ref/(ref+alt)
                #major=alt/len(y)
                #print(MAF)
            MAF_list.append(MAF)    
            j+=3           
    elif(categorical==False):
        if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
            while j < (len(x[0])-1):
                ref = 0
                alt = 0
                MAF = 0       
                for i in range(len(x)):                   
                    ref+=x[i][j]
                    alt+=x[i][j+1]
                    if(x[i][j] != x[i][j+1]):
                        ref+=x[i][j]                     
                        alt+=x[i][j+1]                       
                if(alt<=ref):
                    MAF=alt/(ref+alt)
                    #major=ref/len(y)
                else:
                    MAF=ref/(ref+alt)
                MAF_list.append(MAF)    
                j+=2
        else:
            while j < (len(x[0])-1):
                ref = 0
                alt = 0
                MAF = 0       
                for i in range(len(x)):
                    ref+=x[i][j]
                    alt+=x[i][j+1]   
                if(alt<=ref):
                    MAF=alt/(ref+alt)
                    #major=ref/len(y)
                else:
                    MAF=ref/(ref+alt)
                MAF_list.append(MAF)    
                j+=2
    return MAF_list

def calculate_MAF_global_GPU(indexes, inx, categorical="False"):
    

    j=0
    if(do_parallel_MAF==True):
        getter = operator.itemgetter(indexes)
        x = list(map(list, map(getter, np.copy(inx))))
    else:
        x = inx
    MAF_list = []
        
    #tf.reset_default_graph()
    with tf.compat.v1.Session() as sess:
    #with tf.Session(config=config) as sess:        
       
        #print("LENGTH", len(x[0]))
        if(categorical=="True"):
            while j < (len(x[0])):
                ref = 0
                alt = 0
                MAF = 0        
                for i in range(len(x)):
                    if(x[i][j] == 0):
                        ref = sess.run(tf.add(ref,2))
                    elif(x[i][j] == 1):
                        ref = sess.run(tf.add(ref,1))
                        alt = sess.run(tf.add(alt,1))
                    elif(x[i][j] == 2):
                        alt = sess.run(tf.add(alt,2))
                if(alt<=ref):
                    MAF=sess.run(tf.div(alt,tf.add(ref,alt)))
                    #major=ref/len(y)
                else:
                    MAF=sess.run(tf.div(ref,tf.add(ref,alt)))
                    #major=alt/len(y)
                    #print(MAF)
                MAF_list.append(MAF)
                j+=1          
        elif(categorical=="False"):
            if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
                while j < (len(x[0])):
                    ref = 0
                    alt = 0
                    MAF = 0        
                    for i in range(len(x)):
                        ref = sess.run(tf.add(ref,x[i][j][0]))
                        alt = sess.run(tf.add(alt,x[i][j][1])) 
                        if(x[i][j][0]!=x[i][j][1]):
                            ref = sess.run(tf.add(ref,x[i][j][0]))
                            alt = sess.run(tf.add(alt,x[i][j][1]))
                    if(alt<=ref):
                        MAF=sess.run(tf.div(alt,tf.add(ref,alt)))
                        #major=ref/len(y)
                    else:
                        MAF=sess.run(tf.div(ref,tf.add(ref,alt)))
                    MAF_list.append(MAF)    
                    j+=1            
            else:    
                while j < (len(x[0])):
                    ref = 0
                    alt = 0
                    MAF = 0        
                    for i in range(len(x)):
                        ref = sess.run(tf.add(ref,x[i][j][0]))
                        alt = sess.run(tf.add(alt,x[i][j][1]))  
                    if(alt<=ref):
                        MAF=sess.run(tf.div(alt,tf.add(ref,alt)))
                        #major=ref/len(y)
                    else:
                        MAF=sess.run(tf.div(ref,tf.add(ref,alt)))
                    MAF_list.append(MAF)    
                    j+=1
    
    #reset tensorflow session
    #tf.reset_default_graph()
    sess.close()
    return MAF_list

def calculate_MAF_global(indexes, inx, categorical="False"):
    j=0
    if(do_parallel_MAF==True):
        getter = operator.itemgetter(indexes)
        x = list(map(list, map(getter, np.copy(inx))))
    else:
        x = inx
    MAF_list = []
    #print("LENGTH", len(x[0]))
    if(categorical=="True"):
        while j < (len(x[0])):
            ref = 0
            alt = 0
            MAF = 0        
            for i in range(len(x)):
                if(i == 0):
                    ref+=2
                elif(i == 1):
                    ref+=1
                    alt+=1
                elif(i == 2):
                    alt+=2
            if(alt<=ref):
                MAF=alt/(ref+alt)
                #major=ref/len(y)
            else:
                MAF=ref/(ref+alt)
                #major=alt/len(y)
                #print(MAF)
            MAF_list.append(MAF)    
            j+=1          
    elif(categorical=="False"):
        if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
            while j < (len(x[0])):
                ref = 0
                alt = 0
                MAF = 0
                for i in range(len(x)):
                    ref+=x[i][j][0]
                    alt+=x[i][j][1]
                    if(x[i][j][0]!=x[i][j][1]):
                        ref+=x[i][j][0]
                        alt+=x[i][j][1]
                if(alt<=ref):
                    MAF=alt/(ref+alt)
                    #major=ref/len(y)
                else:
                    MAF=ref/(ref+alt)
                MAF_list.append(MAF)
                j+=1
        else:
            while j < (len(x[0])):
                ref = 0
                alt = 0
                MAF = 0
                for i in range(len(x)):
                    ref+=x[i][j][0]
                    alt+=x[i][j][1]
                if(alt<=ref):
                    MAF=alt/(ref+alt)
                    #major=ref/len(y)
                else:
                    MAF=ref/(ref+alt)
                MAF_list.append(MAF)
                j+=1
    return MAF_list


#TODO review this and make this work with CE, WCE, and FL
def mean_estimated_r2_GPU(x, categorical=False):

    #tf.reset_default_graph()
    
    sess = tf.Session(config=config)    
    
    def calculate_r2hat(x_genotypes, MAF):
    #This calculates exactly the same r2 as the estimated r2hat from minimac4
    #I copy this from Minimac4 source code, exactly as it is in Minimac4
    #Credits to Minimac4 authors
        r2hat=0
        mycount = tf.cast(len(x_genotypes), tf.float64)
        mysum = tf.cast(0, tf.float64)
        mysum_Sq = tf.cast(0, tf.float64)
        
        if(MAF==0): #dont waste time
            #print("r2hat", r2hat)
            return r2hat
        
        for i in range(len(x_genotypes)):
            #print("X", x_genotypes[i])
            if(x_genotypes[i]==0):
                d = 0
            else:                
                d = np.divide(x_genotypes[i],2)
            
            if (d>0.5):
                d = 1-d
                d = tf.cast(d, tf.float64)
            mysum_Sq = sess.run(tf.add(mysum_Sq, tf.multiply(d,d)))
            mysum = sess.run(tf.add(mysum,d))
        
        if(mycount < 2):#return 0
            #print("r2hat", r2hat)
            return r2hat
        myvar = tf.cast(1e-30, tf.float64)
        myf = sess.run(tf.div(mysum,tf.add(mycount,myvar)))
        #print("myf", myf)
        myevar = sess.run( tf.multiply(myf, tf.subtract(1.0,myf) ) )
        #print("myevar", myf)
        
        #mysum_Sq - mysum * mysum / (mycount + 1e-30)
        myovar = sess.run( tf.divide(tf.subtract(mysum_Sq,tf.multiply(mysum,mysum), tf.add(mycount,myvar))) )
        #print("myovar", myovar)

        if(myovar>0):

            myovar = sess.run(tf.divide(myovar, tf.add(mycount,mayvar)))
            r2hat = sess.run(tf.divide(myovar, tf.add(myevar, myvar)))
        
        #print("r2hat", r2hat)

        return r2hat[0]
     
    j=0
    mean_r2hat = 0
    r2hats = []
    #MAFs = calculate_MAF(x, categorical)
    idx = 0
    print(len(x[0]), len(MAF_all_var))
    while j < len(x[0]):
        if(categorical==False):
            getter = operator.itemgetter([j+1])
            x_genotypes = list(map(list, map(getter, np.copy(x))))
            j+=2
        else:
            x_genotypes = []
            for i in range(len(x)):
                x_genotypes.append(np.argmax(x[i][j:j+3]))                
            j+=3
        
        r2hat = calculate_r2hat(x_genotypes, MAF_all_var[idx])
        r2hats.append(r2hat)
        if(r2hat>0):
            mean_r2hat += r2hat/len(MAF_all_var)
        idx += 1 
        if(idx>=len(MAF_all_var)):
            break

    #tf.reset_default_graph()
    sess.close()
                          
    return mean_r2hat, r2hats

#TODO review this and make this work with CE, WCE, and FL
def mean_estimated_r2(x, categorical=False):

    def calculate_r2hat(x_genotypes, MAF):
    #This calculates exactly the same r2 as the estimated r2hat from minimac4
    #I copy this from Minimac4 source code, exactly as it is in Minimac4
    #Credits to Minimac4 authors
        r2hat=0
        mycount = len(x_genotypes)
        mysum = 0
        mysum_Sq = 0
        
        if(MAF==0): #dont waste time
            #print("r2hat", r2hat)
            return r2hat
        
        for i in range(mycount):
            #print("X", x_genotypes[i])
            if(x_genotypes[i]==0):
                d = 0
            else:
                d = np.divide(x_genotypes[i],2)
            
            if (d>0.5):
                d = 1-d
            mysum_Sq += (d*d)
            mysum += d
        
        if(mycount < 2):#return 0
            #print("r2hat", r2hat)
            return r2hat
        
        myf = mysum / (mycount + 1e-30)
        #print("myf", myf)
        myevar = myf * (1.0 - myf)
        #print("myevar", myf)
        myovar = mysum_Sq - mysum * mysum / (mycount + 1e-30)
        #print("myovar", myovar)

        if(myovar>0):

            myovar = myovar / (mycount + 1e-30)
            r2hat = myovar / (myevar + 1e-30)
        
        #print("r2hat", r2hat)

        return r2hat
     
    j=0
    mean_r2hat = 0
    r2hats = []
    #MAFs = calculate_MAF(x, categorical)
    idx = 0
    print(len(x[0]), len(MAF_all_var))
    while j < len(x[0]):
        if(categorical==False):
            getter = operator.itemgetter([j+1])
            x_genotypes = list(map(list, map(getter, np.copy(x))))
            j+=2
        else:
            x_genotypes = []
            for i in range(len(x)):
                x_genotypes.append(np.argmax(x[i][j:j+3]))                
            j+=3
        
        r2hat = calculate_r2hat(x_genotypes, MAF_all_var[idx])
        r2hats.append(r2hat)
        if(r2hat>0):
            mean_r2hat += r2hat/len(MAF_all_var)
        idx += 1 
        if(idx>=len(MAF_all_var)):
            break

    return mean_r2hat, r2hats


def filter_by_MAF(x,y, MAFs, threshold1=0, threshold2=1, categorical=False):
    
    colsum=np.sum(y, axis=0)
    indexes_to_keep = []
    i = 0
    j = 0
    k = 0   
    if(verbose>0):
        print("COOLSUM", len(colsum), ":", colsum[0:13])

    while i < len(MAFs):
        if(MAFs[i]>threshold1 and MAFs[i]<=threshold2):
            if(categorical==True or categorical=="True"):
                if(colsum[j]!=0 or colsum[j+1]!=0 or colsum[j+2]!=0):
                    indexes_to_keep.append(j)
                    indexes_to_keep.append(j+1)
                    indexes_to_keep.append(j+2)
            elif(categorical==False or categorical=="False"):
                if(colsum[k]!=0 or colsum[k+1]!=0):
                    indexes_to_keep.append(k)
                    indexes_to_keep.append(k+1)
                else:
                    print("WARNING!!!!! INDEX", i, "HAS good MAF but colsum is zero")            
        i += 1
        j += 3
        k += 2
    if(verbose>0):
        print("FILTER BY MAF INDEXES TO KEEP:", len(indexes_to_keep), indexes_to_keep)

    getter = operator.itemgetter(indexes_to_keep)

    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
    
    return filtered_data_x, filtered_data_y

def accuracy_maf_threshold(sess, x, y, MAFs, threshold1=0, threshold2=1, categorical=False):
    

    filtered_data_x, filtered_data_y = filter_by_MAF(x,y, MAFs, threshold1, threshold2, categorical)
    
    correct_prediction = np.equal( np.round( filtered_data_x ), np.round( filtered_data_y ) )
    accuracy_per_marker = np.mean(correct_prediction.astype(float), 0)
    accuracy = np.mean(accuracy_per_marker)

    #correct_prediction = sess.run(tf.equal( tf.round( filtered_data_x ), tf.round( filtered_data_y ) ))
    #accuracy_per_marker = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 0))
    #accuracy = sess.run(tf.reduce_mean(accuracy_per_marker))

    return accuracy, accuracy_per_marker

def MSE_maf_threshold(sess, x, y, MAFs, threshold1=0, threshold2=1, categorical=False):
    
    filtered_data_x, filtered_data_y = filter_by_MAF(x,y, MAFs, threshold1, threshold2, categorical)
    
    MSE_per_marker = np.mean(np.square( np.subtract( np.round( filtered_data_x ), np.round( filtered_data_y ) ) ), 0 ) 
    MSE = np.mean( MSE_per_marker )
    
    #MSE_per_marker = sess.run( tf.reduce_mean(tf.square( tf.subtract( tf.round( filtered_data_x ), tf.round( filtered_data_y ) ) ), 0 ) )
    #MSE = sess.run( tf.reduce_mean( MSE_per_marker ) )

    return MSE, MSE_per_marker

def accuracy_maf_threshold_global(sess, x, y, indexes_to_keep):
    
    getter = operator.itemgetter(indexes_to_keep)
    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
       
    correct_prediction = np.equal( np.round( filtered_data_x ), np.round( filtered_data_y ) )
    accuracy_per_marker = np.mean(correct_prediction.astype(float), 0)
    accuracy = np.mean(accuracy_per_marker)

    #correct_prediction = sess.run(tf.equal( tf.round( filtered_data_x ), tf.round( filtered_data_y ) ))
    #accuracy_per_marker = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 0))
    #accuracy = sess.run(tf.reduce_mean(accuracy_per_marker))

    return accuracy, accuracy_per_marker

def MSE_maf_threshold_global(sess, x, y, indexes_to_keep):
    
    getter = operator.itemgetter(indexes_to_keep)
    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
    
    MSE_per_marker = np.mean(np.square( np.subtract( np.round( filtered_data_x ), np.round( filtered_data_y ) ) ), 0 ) 
    MSE = np.mean( MSE_per_marker )
    
    #MSE_per_marker = sess.run( tf.reduce_mean(tf.square( tf.subtract( tf.round( filtered_data_x ), tf.round( filtered_data_y ) ) ), 0 ) )
    #MSE = sess.run( tf.reduce_mean( MSE_per_marker ) )

    return MSE, MSE_per_marker



def flatten_data(sess, myinput, factors=False):
    x = np.copy(myinput)
    if(factors == False): #if shape=3 we are using allele count instead of factors
        x = np.reshape(x, (x.shape[0],-1))
    else:#do one hot encoding, depth=3 because missing (-1) is encoded to all zeroes
        x = (tf.one_hot(indices=x, depth=3))
        x = (tf.layers.flatten(x))#flattening to simplify calculations later (matmul, add, etc)
        x = x.eval()
        #print("Dimensions after flatting", x.shape)
    return x


def flatten_data_np(x):
    #x = np.copy(myinput)
    x = np.reshape(x, (x.shape[0],-1))
    return x

def define_weights(ni,nh,no,last_layer=False):
    w = {
        'encoder_h1': tf.Variable(tf.random_normal([ni, nh], dtype=tf.float64), name="w_encoder_h1"),
    }
    if(last_layer==True):
        w['decoder_h1']=tf.Variable(tf.random_normal([nh, no], dtype=tf.float64), name="w_decoder_h1")
    return w

def define_biases(nh,no,last_layer=False):
    
    b = {
        'encoder_b1': tf.Variable(tf.random_normal([nh], dtype=tf.float64), name="b_encoder_b1"),
    }
    if(last_layer==True):
        b['decoder_b1']=tf.Variable(tf.random_normal([no], dtype=tf.float64), name="b_decoder_b1")
    return b
#Code modified from example
#https://stackoverflow.com/questions/44367010/python-tensor-flow-relu-not-learning-in-autoencoder-task
def run_autoencoder(learning_rate, training_epochs, l1_val, l2_val, act_val, beta, rho, data_obs):

    prep_start = timeit.default_timer()
    
    #custom beta will scale sparsity loss proportional to the number of features
    if(custom_beta==True and beta>0):
        beta = tf.cast(beta*data_obs.shape[1], tf.float64)
    #otherwise, just scale sparsity loss by using beta as an absolute scaling value
    else:
        beta = tf.cast(beta, tf.float64)
    
    print("Running autoencoder.")
    
    factors = True
    #subjects, SNP, REF/ALT counts
    if(len(data_obs.shape) == 3):
        factors = False

    print("Input data shape:")
    #print(data_masked.shape)
    print(data_obs.shape)
    
    original_shape = data_obs.shape
    
    batch_size = int(round(len(data_obs)/split_size)) # size of training objects split the dataset in 10 parts
    #print(batch_size)
    
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
        #n_hidden_1 = int(round(n_output*hsize))  # hidden layer for encoder, equal to input number of features multiplied by a hidden size ratio
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
               
    #encoder_result = encoder_op
    y_pred = tf.identity(y_pred, name="y_pred")

    #print(encoder_op)
    # predict result
    #y_pred = decoder_op
    # real input data as labels
    y_true = Y
    #TODO experimental Maximal information criteria calculation needs to be implemented
    #M = tf.zeros([n_input, n_input], name="MIC")
   
    rho_hat = tf.reduce_mean(encoder_operators[1],0) #RR sometimes returns Inf in KL function, caused by division by zero, fixed wih logfun()
    if (act_val == "tanh"):
        rho_hat = tf.div(tf.add(rho_hat,1.0),2.0) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl
    if (act_val == "relu" or act_val == "relu_tanh" or act_val == "relu,tanh"):
        rho_hat = tf.div(tf.add(rho_hat,1e-10),tf.reduce_max(rho_hat)) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl

    if(NH>=2 and all_sparse==True):
        rho_hat2 = tf.reduce_mean(encoder_operators[2],0) #RR sometimes returns Inf in KL function, caused by division by zero, fixed wih logfun()
        if (act_val == "tanh"):
            rho_hat2 = tf.div(tf.add(rho_hat2,1.0),2.0) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl
        if (act_val == "relu" or act_val == "relu_tanh" or act_val == "relu,tanh"):
            rho_hat2 = tf.div(tf.add(rho_hat2,1e-10),tf.reduce_max(rho_hat2)) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl
    if(NH==3 and all_sparse==True):
        rho_hat3 = tf.reduce_mean(encoder_operators[2],0) #RR sometimes returns Inf in KL function, caused by division by zero, fixed wih logfun()
        if (act_val == "tanh"):
            rho_hat3 = tf.div(tf.add(rho_hat3,1.0),2.0) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl
        if (act_val == "relu" or act_val == "relu_tanh" or act_val == "relu,tanh"):
            rho_hat3 = tf.div(tf.add(rho_hat3,1e-10),tf.reduce_max(rho_hat3)) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl
    
    #rho = tf.constant(rho) #not necessary maybe?

    with tf.name_scope('sparsity'):
        sparsity_loss = tf.reduce_mean(KL_Div(rho, rho_hat))
        sparsity_loss = tf.cast(sparsity_loss, tf.float64)
        #sparsity_loss = tf.clip_by_value(sparsity_loss, 1e-10, 1.0) #RR KL divergence, clip to avoid Inf or div by zero

        if(NH>=2 and all_sparse==True):
            sparsity_loss2 = tf.cast(tf.reduce_mean(KL_Div(rho, rho_hat2)), tf.float64)
            #sparsity_loss2 = tf.clip_by_value(sparsity_loss2, 1e-10, 1.0) #RR KL divergence, clip to avoid Inf or div by zero
            sparsity_loss = tf.add(sparsity_loss, sparsity_loss2)

        if(NH==3 and all_sparse==True):
            sparsity_loss3 = tf.cast(tf.reduce_mean(KL_Div(rho, rho_hat3)), tf.float64)
            sparsity_loss = tf.add(sparsity_loss, sparsity_loss3)
            
        sparsity_loss = tf.cast(sparsity_loss, tf.float64, name="sparsity_loss") #RR KL divergence, clip to avoid Inf or div by zero

    tf.summary.scalar('sparsity_loss', sparsity_loss)

    # define cost function, optimizers
    # loss function: MSE # example cost = tf.reduce_mean(tf.square(tf.subtract(output, x)))
    with tf.name_scope('loss'):

        if(loss_type=="MSE"):
            y_true = tf.cast(y_true, tf.float64)
            reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred)), name="reconstruction_loss") #RR simplified the code bellow
        elif(loss_type=="Pearson"):
            reconstruction_loss = pearson_r_loss(y_pred, y_true)
        elif(loss_type=="WMSE"):
            reconstruction_loss = weighted_MSE(y_pred, y_true)
        elif(loss_type=="CE"):
            reconstruction_loss = cross_entropy(y_pred, y_true)
        elif(loss_type=="WCE"):
            reconstruction_loss = weighted_cross_entropy(y_pred, y_true)
        elif(loss_type=="FL"):
            reconstruction_loss = focal_loss(y_pred, y_true)
            if(verbose>0):
                mygamma_0, mygamma_1 = calculate_gamma(y_pred, y_true)
                ce0, ce1 = calculate_CE(y_pred, y_true)
                pt0, pt1 = calculate_pt(y_pred, y_true)
                wce = weighted_cross_entropy(y_pred, y_true)
        else:
            y_true = tf.cast(y_true, tf.float64)            
            reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred)), name="reconstruction_loss") #RR 
        # newly added sparsity function (RHO, BETA)
        #cost_sparse = tf.multiply(beta,  tf.reduce_sum(KL_Div(RHO, rho_hat)))
        #print(cost_sparse) 

        cost = tf.reduce_mean(tf.add(reconstruction_loss,tf.multiply(beta, sparsity_loss)), name = "cost") #RR simplified equation
        
    tf.summary.scalar('reconstruction_loss_MSE', reconstruction_loss)
    tf.summary.scalar("final_cost", cost)

    #TODO: add a second scaling factor for MAF loss, beta2*MAF_loss
    #or add MAF as another feature instead of adding it to the loss function
    #p-value for fisher exact test or similar type of test as MAF_loss
    #MAF_loss = 0
    #for i in range(0, len(y_pred[0])):
    #        MAF_loss = MAF_loss + stat.fisher(y_pred[][i],y_true[][i])
    #        
    #cost = tf.reduce_mean(reconstruction_loss + beta * sparsity_loss + beta2 * MAF_loss)
    #cost = tf.reduce_mean(cost + cost_sparse)
    
    correct_prediction = 0
    
    if(factors == False): #TODO: only factors==False is working now, TODO: fix the else: bellow
        y_true = tf.cast(y_true, tf.float64)
        correct_prediction = tf.equal( tf.round( y_pred ), tf.round( y_true ) )
    else:
        y_pred_shape = tf.shape(y_pred)
        y_true_shape = tf.shape(y_true)   
        #new_shape = [tf.cast(y_pred_shape[0], tf.int32), tf.cast(n_input/3,  tf.int32), tf.cast(3, tf.int32)]
        #a = tf.cast(y_pred_shape[0], tf.int32)
        #print("shape for calculating accuracy:", a.eval())
        #print(new_shape.eval())
        #new_shape = [tf.shape(y_true_shape)[0],original_shape[1], 3]
        #reshaping back to original form, so the argmax function can work
        y_pred_tmp = tf.reshape(y_pred, [tf.shape(y_true_shape)[0],original_shape[1], 3]) 
        y_true_tmp = tf.reshape(y_true, [tf.shape(y_true_shape)[0],original_shape[1], 3])
        y_pred_tmp = tf.argmax( y_pred_tmp, 1 ) #back to original categorical form [-1,0,1,2]
        y_true_tmp = tf.argmax( y_true_tmp, 1 ) #back to original categorical form [-1,0,1,2]
        #y_true_tmp = tf.argmax( y_true, 1 ) #back to original categorical form [-1,0,1,2]
        #y_pred_tmp = tf.argmax( y_pred, 1 ) #back to original categorical form [-1,0,1,2]        
     
        correct_prediction = tf.equal( y_pred_tmp, y_true_tmp )
        #correct_prediction = tf.equal( y_pred, y_true )
                
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64), name="accuracy")
    cost_accuracy = tf.add((1-accuracy), tf.multiply(beta, sparsity_loss), name="cost_accuracy")
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('cost_accuracy', cost_accuracy)
    
    #The RMSprop optimizer is similar to the gradient descent algorithm with momentum. 
    #The RMSprop optimizer restricts the oscillations in the vertical direction. 
    #Therefore, we can increase our learning rate and our algorithm could take larger steps in the horizontal direction converging faster. 
    #ref: https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b
    #Converge by accuracy when using factors
    if(recovery_mode=="False"):
        with tf.name_scope('train'):
            if(optimizer_type=="RMSProp"):
                if(factors == False):
                    optimizer = tf.train.RMSPropOptimizer(learning_rate, name="optimizer").minimize(cost)
                else:
                    optimizer = tf.train.RMSPropOptimizer(learning_rate, name="optimizer").minimize(cost_accuracy)
            elif(optimizer_type=="GradientDescent"):
                if(factors == False):
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate, name="optimizer").minimize(cost)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate, name="optimizer").minimize(cost_accuracy)
            elif(optimizer_type=="Adam"):
                if(factors == False):
                    optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(cost)
                else:
                    optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(cost_accuracy)
        print("Optimizer:", optimizer)
        #save this optimizer to restore it later.
        #tf.add_to_collection("optimizer", optimizer)
        #tf.add_to_collection("Y", Y)
        #tf.add_to_collection("X", X)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # initialize variables
    #init = tf.global_variables_initializer();
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    start = timeit.default_timer()
    r_report_time = 0
    mask_time = 0
    time_metrics = 0
    gd_time = 0
    # run autoencoder .........
   
    #with tf.device('/device:GPU:0'):  # Replace with device you are interested in
    #    bytes_in_use = BytesInUse()
    
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
            '''
            print("Global collection:", global_name)
            print("Local collection:", local_name)
            print("Trainable:", trainable_vars)
            for i in tf.get_collection(tf.compat.v1.global_variables, scope='train'):
                print("train:",i)
            for i in tf.get_collection(tf.compat.v1.global_variables, scope='train'):
                print("train:",i)
            for i in tf.get_collection(tf.compat.v1.global_variables, scope='train'):
                print("train:",i)
            for i in tf.compat.v1.get_collection(tf.compat.v1.global_variables, scope='train'):
                print("train1:",i)
            for i in tf.compat.v1.get_collection(tf.compat.v1.global_variables, scope='train'):
                print("train1:",i)
            for i in tf.compat.v1.get_collection(tf.compat.v1.global_variables, scope='train'):
                print("train1:",i)
            a=[n.name for n in graph.as_graph_def().node]
            print("node names:", a)

            print("scope:::", tf.compat.v1.get_default_graph().get_name_scope())
            all_ops = graph.get_operations()
            for el in all_ops:
                print(el)
            ''' 
            #optimizer = graph.get_tensor_by_name("optimizer:0")

            a=[n.name for n in graph.as_graph_def().node]
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
            sparsity_loss = graph.get_tensor_by_name("sparsity/sparsity_loss:0")
            accuracy = graph.get_tensor_by_name("accuracy:0")

            #accuracy = "accuracy:0"
            #cost_accuracy = "cost_accuracy:0"
            cost_accuracy = graph.get_tensor_by_name("cost_accuracy:0")
            y_pred = graph.get_tensor_by_name("y_pred:0")

            tf.summary.scalar('reconstruction_loss_MSE', reconstruction_loss)
            tf.summary.scalar("final_cost", cost)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('cost_accuracy', cost_accuracy)
            tf.summary.scalar('sparsity_loss', sparsity_loss)
            
            weights = {
                'encoder_h1': graph.get_tensor_by_name("weights/w_encoder_h1:0"),
                'decoder_h1': graph.get_tensor_by_name("weights/w_decoder_h1:0"),
            }
            variable_summaries(weights['encoder_h1'])
        
            biases = {
                'encoder_b1': graph.get_tensor_by_name("biases/b_encoder_b1:0"),
                'decoder_b1': graph.get_tensor_by_name("biases/b_decoder_b1:0"),
            }
            variable_summaries(biases['encoder_b1'])
            
            decoder_op = y_pred
            #encoder_op = graph.get_tensor_by_name("Wx_plus_b/activation:0")
            
            encoder_op =  graph.get_tensor_by_name("Wx_plus_b/dense/BiasAdd:0")

            tf.summary.histogram('activations', encoder_op)

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
            data_masked = mask_data_per_sample_parallel([0],np.copy(data_obs),mask_rate,categorical)

            mask_stop = timeit.default_timer()
            print("Time to run masking: ", mask_stop-mask_start)
            mask_time += mast_stop-mask_start

            if(disable_masking==False):
                data_masked = flatten_data(sess, data_masked, factors)
                
            data_obs = flatten_data(sess, data_obs, factors)

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
                train_args = [merged,optimizer, cost, accuracy, reconstruction_loss, mygamma_0, mygamma_1, ce0, ce1,pt0, pt1, wce, y_pred]
            else:
                train_args = [merged,optimizer, cost, accuracy, reconstruction_loss, y_pred]
        else:
            if(verbose>1):
                train_args = [optimizer, cost, accuracy, reconstruction_loss, mygamma_0, mygamma_1, ce0, ce1, pt0, pt1, wce, y_pred]
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

            if(mask_rate>maximum_masking_rate):
                mask_rate=maximum_masking_rate

            if(cycle_count==repeat_cycles):
                cycle_count = 0
            else:
                cycle_count += 1

            #make new masking on every new iteration
            if( (epoch>=0 and do_parallel_gd_and_mask==False) or epoch==0):
                if(do_parallel_numpy_per_cycle == False):
                    data_masked = mask_data_per_sample_parallel([0], np.copy(data_obs), mask_rate, categorical)
                if(do_parallel_numpy_per_cycle==True):
                    if(cycle_count==0):
                        m_result, pool = submit_mask_tasks(np.copy(data_obs), mask_rate, categorical)
                        data_masked_replicates = retrieve_cycle_results(m_result,pool)
                        
                    data_masked = data_masked_replicates[cycle_count]
                
                if(validate_after_epoch==True):
                    data_val_masked = mask_data_per_sample_parallel([0], np.copy(data_obs),maximum_masking_rate,categorical)
                    val_x = data_val_masked
                if(verbose>0):
                    print("Mask done for epoch",iepoch,"Result length:",len(data_masked_replicates), "Data masked shape:", data_masked.shape)
                    
            train_x = np.copy(data_masked)
            train_y = np.copy(data_obs)                               
            #if(epoch>=0 and do_parallel_gd_and_mask==False or (epoch==0 and do_parallel_gd_and_mask==True)):
                
            mask_stop = timeit.default_timer()
            mask_time += mask_stop-mask_start

            flat_start = timeit.default_timer()
            #after masking, flatten data
            #train_x = flatten_data(sess, train_x, factors)
            #train_y = flatten_data(sess, train_y, factors)
            train_x = flatten_data_np(train_x)
            train_y = flatten_data_np(train_y)
            
            if(validate_after_epoch==True):
                val_x = flatten_data(sess, val_x, factors)

            if(n_input!=n_output):
                train_y = list(map(list, map(o_getter, np.copy(train_y))))

            flat_stop = timeit.default_timer()
            flat_time += flat_stop-flat_start

            shuf_start = timeit.default_timer()
            if(shuffle==True):
                #randomize = np.arange(len(train_x))
                randomize = np.random.rand(len(train_x)).argsort()
                #np.random.shuffle(randomize)
                #train_x = np.asarray(train_x)
                #train_y = np.asarray(train_y)
                train_x = train_x[randomize]
                train_y = train_y[randomize]
                if(validate_after_epoch==True):
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
                
                    #data_masked_replicates, my_result_list = run_parallel_gd_and_mask(sess, X, Y, data_masked, data_obs, mask_rate, categorical, train_args, batch_size)
                    #faster to call directly from main process
                    mask_start = timeit.default_timer()
                    m_result, pool = submit_mask_tasks(data_obs, mask_rate, categorical)
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

                if(validate_after_epoch==True):
                    batch_val_x = val_x[i*batch_size:(i+1)*batch_size]
                    
                rest_stop = timeit.default_timer()
                rest_time += rest_stop-rest_start
                #calculate cost and optimizer functions                    
                if(i!=(total_batch-1) or last_batch_validation==False):
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
                            summary, _, c, a, rl, g0, g1, myce0, myce1,mypt0, mypt1, mywce, myy = my_result
                        else:
                            summary, _, c, a, rl, myy = my_result
                        train_writer.add_run_metadata(run_metadata, 'k%03d-step%03d-batch%04d' % (ki, epoch, i) )
                        train_writer.add_summary(summary, epoch)
                    else:
                        if(verbose>1):
                            _, c, a, rl, g0, g1, myce0, myce1,mypt0, mypt1, mywce, myy = my_result
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

                if(validate_after_epoch==True):
                    my_val_pred = vc, va, vrl, vmyy  = sess.run([cost, accuracy, reconstruction_loss, y_pred], feed_dict={X: batch_val_x, Y: batch_y} )    
                    my_vr2_tmp = pearson_r2(vmyy, batch_y)
                    val_avg_r2 += np.sum(my_vr2_tmp['r2_per_sample'])/len(my_vr2_tmp['r2_per_sample'])/total_batch
                    val_avg_a += va/total_batch
                    val_avg_rl += vrl/total_batch
                    val_epoch_cost+=vc/total_batch
                    val_avg_cost+=vc/total_batch/window
                    
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
                mask_rate
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
                print("Epoch ", epoch, " done. Masking rate used:", mask_rate, " Initial one:", initial_masking_rate, " Loss: ", epoch_cost, " Accuracy:", avg_a, " Reconstruction loss (" , loss_type, "): ", avg_rl, "ce0:", myce0, "ce1:", myce1, "sr2:", avg_r2, flush=True)
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
                print("Epoch ", epoch, " done. Masking rate used:", mask_rate, " Initial one:", initial_masking_rate, " Loss: ", epoch_cost, " Accuracy:", avg_a, " s_r2:", avg_r2, " Reconstruction loss (" , loss_type, "): ", avg_rl, flush=True)
            if(validate_after_epoch==True):
                print("VALIDATION:Epoch ", epoch, " done. Masking rate used:", mask_rate, " Initial one:", initial_masking_rate, " Loss: ", val_epoch_cost, " Accuracy:", val_avg_a, " s_r2:", val_avg_r2, " Reconstruction loss (" , loss_type, "): ", val_avg_rl, flush=True)

            
            rest_stop = timeit.default_timer()
            rest_time += rest_stop-rest_start  
                            
            stop_epochs = timeit.default_timer()
            time_epochs += stop_epochs-start_epochs
            
            save_start = timeit.default_timer()
            
            if(save_model==True and iepoch>0 and (iepoch==training_epochs or epoch+1 % window == 0) ):
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
                #print(i,total_batch,my_r2_tmp['r2_per_sample'])
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
                #my_acc_t[0], _ = accuracy_maf_threshold(sess, my_pred, val_y, MAF_all_var, 0.005, 1, factors)
                #my_acc_t[1], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0, 0.005, factors)
                my_acc_t[2], _ = accuracy_maf_threshold(sess, my_pred, train_y, MAF_all_var, 0, 0.01, factors)
                my_acc_t[3], _ = accuracy_maf_threshold(sess, my_pred, train_y, MAF_all_var, 0.01, 1, factors) 
            else:
                my_acc_t[0], my_acc_t[1],my_acc_t[2],my_acc_t[3] = "NA","NA", "NA", "NA"

        r_report_start = timeit.default_timer()

        print("Accuracy [MAF 0-0.01]:", my_acc_t[2])
        print("Accuracy [MAF 0.01-1]:", my_acc_t[3])
        print("F1 score [MAF 0-1]:", my_F1)
        print("R-squared per sample:", my_r2)

        r_report_stop = timeit.default_timer()
        print("Time to calculate accuracy (rare versus common variants:", r_report_stop-r_report_start)
        r_report_time += r_report_stop-r_report_start

        if(detailed_metrics==True):

            #my_pred = sess.run([y_pred], feed_dict={X: train_x, Y: train_y})
            my_pred = sess.run([y_pred], feed_dict={X: train_x, Y: train_y})
            my_pred = my_pred[0]
            print("Accuracy per veriant...")
            my_acc_t[0], acc_per_m = accuracy_maf_threshold(sess, my_pred, train_y, MAF_all_var, 0, 1, factors)

            print("MSE per veriant...")
            my_MSE, MSE_list = MSE_maf_threshold(sess, my_pred, train_y, MAF_all_var, 0, 1, factors)

            print("Estimated r2hat per veriant...")
            mean_r2hat_est, my_r2hat_est = mean_estimated_r2(train_y, factors)
            print("Emprirical r2hat per veriant...")
            mean_r2hat_emp, my_r2hat_emp = mean_empirical_r2(train_x, train_y, factors)


        #print("MAF", len(my_MAF_list), "acc", len(acc_per_m), "r2emp", len(my_r2hat_emp), "r2est", len(my_r2hat_emp), "MSE", len(MSE_list))
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

    if(len(sys.argv)!=6 and len(sys.argv)!=7):
        return False
    
    if(save_model==True):
        model_dir=os.path.basename(sys.argv[1])+"_model"
        if(os.path.exists(model_dir)==False):
            os.mkdir(model_dir)

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

    if(len(sys.argv)==6 or len(sys.argv)==7):
        print("Parsing input file: ")
            #Arguments
            #sys.argv[1] = [str] input file (HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf)
            #sys.argv[2] = [str] hyperparameter list file (mylist.txt)
            #sys.argv[3] = [True,False] Recovery mode, default is False
            #sys.argv[4] = [1/846] Initial masking rate
            #sys.argv[5] = [0.98] Final masking rate
            #sys.argv[6] = [str] only needed if recovery mode is True, path to the .ckpt file

        recovery_mode = sys.argv[3]
        #If enabling recovery mode to resume training, provide model path after masking rates, as last parameter
        if(recovery_mode=="True"):
            #path to the .ckpt file, example: ./recovery/inference_model-1.ckpt
            model_path=sys.argv[6]

        initial_masking_rate = convert_to_float(sys.argv[4])
        maximum_masking_rate = convert_to_float(sys.argv[5])

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
        print("reading hyperparameter list from file: ", sys.argv[2])

        hp_array = []
        result_list = []

        with open(sys.argv[2]) as my_file:
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
                data_obs = process_data(sys.argv[1],categorical) #input file, i.e: HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf

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
            
    else:   
       
        return False



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
