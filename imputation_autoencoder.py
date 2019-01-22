
# coding: utf-8

# In[ ]:
#latest update: implemented online data augmentation mode / online machine learning mode

#example
#/bin/python3.6 script_name.py imputed_test_subset.vcf 3_hyper_par_set.txt True
#CUDA_VISIBLE_DEVICES=0 /bin/python3.6 ../script_name.py imputed_test_subset.vcf 3_hyper_par_set.txt True
#CUDA_VISIBLE_DEVICES=1 /bin/python3.6 ../script_name.py imputed_test_subset.vcf 0.1 0.01 2 0.7 relu 0.001 False
import math #sqrt

import tensorflow as tf

from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import itertools

import pandas as pd

import random #masking

# sorting results
from collections import defaultdict
from operator import itemgetter

import timeit #measure runtime

from tqdm import tqdm # progress bar

from scipy.stats import pearsonr #remove this, nan bugs are not corrected

from tensorflow.python.client import device_lib #get_available_devices()

from joblib import Parallel, delayed #parallel tasks
import multiprocessing #parallel tasks

from sklearn.model_selection import KFold

from scipy.stats import linregress
#linregress(a, b)[2] gives the correlation, correcting for nans due to monomorphic predictions
#linregress(a, b)[3] gives the p-value correl 
#a are all the predicted genotypes for one given SNP, in ALT dosage form, b is the same but for experimental SNPs

import sys #arguments
import operator #remove entire columns from 2d arrays

###################################OPTIONS#############################################

#backup options
save_model = True #[True,False] save final model generated after all training epochs, generating a checkpoint that can be loaded and continued later
save_pred = False #[True,False] save predicted results for each training epoch and for k-fold CV

#Learning options
split_size = 10 #number of batches
my_keep_rate = 1 #keep rate for dropout funtion, 1=disable dropout
kn = 1 #number of k for k-fold CV (kn>=2); if k=1: just run training
training_epochs = 761 #learning epochs (if fixed masking = True) or learning permutations (if fixed_masking = False), default 500, number of epochs or data augmentation permutations (in data augmentation mode when fixed_masking = False)
#761 permutations will start masking 1 marker at a time, and will finish masking 90% of markers

#Masking options
replicate_ARIC = False #True: use the same masking pattern as ARIC genotype array data, False: Random masking
fixed_masking = False #True: mask variants only at the beggining of the training cycle, False: mask again with a different pattern after each iteration (data augmentation mode)
mask_per_sample = True #True: randomly mask genotypes per sample instead of mask the entire variant for all samples, False: mask the entire variant for all samples 
random_masking = True #set random masks instead of preset ones
disable_masking = False #disable masking completly, just for learning the data structure in the grid search
initial_masking_rate = 1/846 #when random_masking = False begin masking one variant at a time and gradually increase one by one, not random, 9p21.3 case is 1/846 for incrementing one variant at a time
mask_preset = False #True: mask from genotype array

#Recovery options, online machine learning mode
recovery_mode = False #False: start model from scratch, True: Recover model from last checkpoint
#path to restore model, in case recovery_mode = True 
model_path = '/home/rdias/myscripts/raqueld/Autoencoder_tensorflow/test_data_augmentation/inference_model-1.ckpt'

###################################OPTIONS#############################################
    
# In[ ]:

# parsing vcf files, 
#If categorical false: convert 1|1 -> [0,2], 1|0 or 0|1 -> [1,1], 0|0 -> [2,0], missing -> [0,0]
#If categorical true: convert 1|1 -> 2, 1|0 or 0|1 -> 1, 0|0 -> 0, missing -> -1
#categorical=false is the only mode fully suported now
#todo: finish implenting support for categorical (onehot encoding), just for comparison purposes

def process_data(file, categorical="False"):
    start = timeit.default_timer()
    #Header and column names start with hastag, skip those
    df = pd.read_csv(file, sep='\t', comment='#',header=None)
    #genetic variants are rows and samples are columns
    #let's transpose so the variants become columns and samples are rows
    df_T = df.transpose()
    #print(df.shape) #RR rows are SNPs, subjects are columns
    #print(df_T.shape) #RR rows are Subjects, columns are SNPs
    
    print("This file contains {} features (SNPs) and {} samples (subjects)".format(len(df), (len(df_T)-9)))
    #RR subtract 9 from the number of columns to get the number of SNPs, 
    #RR because the first 9 columns are variant information, not genotypes

    #new_df = [[0] * len(df) for i in range(len(df_T)-8)]
    new_df = 0
    if(categorical=="False"):
        new_df = np.zeros((len(df_T)-9,len(df),2))
    else:
        new_df = np.zeros((len(df_T)-9,len(df)))

    #print(new_df.shape)
    i = 9 #RR column index
    j = 0 #RR row index
    idx = 0
    print("Processing input data.")
    print(categorical)

    #pbar = tqdm(total = len(df_T)-i)
    while i < len(df_T):
        j = 0
        while j < len(df): #"|" is present when phased data is proved, "/" is usually unphased
            if(df[i][j].startswith('1|1') or df[i][j].startswith('1/1')):
                if(categorical=="True"):
                    new_df[idx][j] = 2
                else:
                    #new_df[idx][j] = np.array([0,2])
                    new_df[idx][j][0] = 0
                    new_df[idx][j][1] = 2
            elif(df[i][j].startswith('1|0') or df[i][j].startswith('0|1') or df[i][j].startswith('1/0') or df[i][j].startswith('0/1')):
                if(categorical=="True"):
                    new_df[idx][j] = 1
                else:
                    #new_df[idx][j] = np.array([1,1])
                    new_df[idx][j][0] = 1
                    new_df[idx][j][1] = 1
            elif(df[i][j].startswith('0|0') or df[i][j].startswith('0/0')):
                if(categorical=="True"): 
                    new_df[idx][j] = 0
                else:
                    #new_df[idx][j] = np.array([2,0])
                    new_df[idx][j][0] = 2
                    new_df[idx][j][1] = 0
            else:
                if(categorical=="True"):
                    new_df[idx][j] = -1
                else:
                    #new_df[idx][j] = np.array([0,0]) 
                    new_df[idx][j][0] = 0 
                    new_df[idx][j][1] = 0 
                #RR I forgot to mention that we have to take into account possible missing data
                #RR in case there is missing data (NA, .|., -|-, or anything different from 0|0, 1|1, 0|1, 1|0) = 3
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
    #print(new_df.shape)
    #pbar.close()
    stop = timeit.default_timer()
    print('Time to load the data (sec): ', stop - start)
    return new_df


# In[ ]:


def mask_data(mydata, mask_rate=0.9, categorical="False"):
    start = timeit.default_timer()
    
    #def duplicate_samples(mydata, n):
    #    i=1
    #    while i < n:
    #        mydata.append(mydata)
    #        i+=1
    #    return mydata
    # random matrix the same shape of your data
    #print(len(mydata))
    if(disable_masking == True):
        print("No masking will be done for this run... Just learning data structure")
        return mydata, mydata
    original_data = np.copy(mydata)
    
    def do_masking(mydata,maskindex):     
        #print("Masking markers...")
        #print(maskindex)
        
        #print(mydata.shape)
        
        for i in maskindex:
            #print(len(mydata[i]))
            j = 0
            while j < len(mydata):
                if(categorical=="True"):
                    mydata[j][i]=-1
                else:
                    mydata[j][i]=[0,0]
                j=j+1
    
  
        return mydata    
    
    nmask = int(round(len(mydata[0])*mask_rate))
    # random mask for which values will be changed
    if(random_masking == True):
        maskindex = random.sample(range(0, len(mydata[0]-1)), nmask)
        masked_data = do_masking(np.copy(original_data),maskindex)
        stop = timeit.default_timer()
        #print('Time to mask the data (sec): ', stop - start)
        return masked_data, original_data
    elif(mask_preset==True):    
        maskindex = pd.read_csv('masking_pattern.txt', sep=',', comment='#',header=None)
        # export pandas obj as a list of tuples
        maskindex = [tuple(x) for x in maskindex.values]
        npatterns = len(maskindex)
        for i in range(npatterns):
            maskindex_i = list(maskindex[i])
            masked_data_tmp = do_masking(np.copy(original_data),maskindex_i)
            if(i==0):
                masked_data = np.copy(masked_data_tmp)
                unmasked_data = np.copy(original_data)
            else:
                unmasked_data.append(original_data)
                masked_data.append(masked_data_tmp)
            if(i==(npatterns-1)):
                stop = timeit.default_timer()
                print('Time to mask the data (sec): ', stop - start)
                return masked_data, unmasked_data
    else: #gradual masking in ascending order
        if(categorical=="False"):
            MAFs = calculate_MAF(original_data, False)
        else:
            MAFs = calculate_MAF(original_data, True)
        original_indexes = range(len(MAFs))
        myd = dict(zip(original_indexes,MAFs))
        myd_sorted = sorted(myd.items(), key=itemgetter(1)) #0 keys, 1, values
        sorted_indexes = list(myd_sorted.keys())
        mask_index = sorted_indexes[0:nmask+1]
        masked_data = do_masking(np.copy(original_data),maskindex)
        return masked_data, original_data
    

# In[ ]:
def mask_data_per_sample(mydata, mask_rate=0.9, categorical="False"):
    start = timeit.default_timer()
    # random matrix the same shape of your data
    #print(len(mydata))
    nmask = int(round(len(mydata[0])*mask_rate))
    # random boolean mask for which values will be changed
        
    #np.random.randint(0,len(mydata[0]),size=nmask)
    print("Masking markers...")
    #print(maskindex)
    #mydata = np.transpose(mydata)
    print(mydata.shape)
    #mydata
    
    #pbar = tqdm(total = len(maskindex))
    #for i in range(10):
    #    print(mydata[i][0:11])
    j = 0
    while j < len(mydata):
        #redefines which variants will be masked for every new sample
        maskindex = random.sample(range(0, len(mydata[0]-1)), nmask) 

        for i in maskindex:
            if(categorical=="True"):
                mydata[j][i]=-1
            else:
                mydata[j][i]=[0,0]
        j=j+1

    stop = timeit.default_timer()
    print('Time to mask the data (sec): ', stop - start)  
    return mydata


def logfunc(x, x2):
    cx = tf.clip_by_value(x, 1e-10, 1.0-1e-10)
    cx2 = tf.clip_by_value(x2, 1e-10, 1.0-1e-10)
    return tf.multiply( x, tf.log(tf.div(cx,cx2)))


#Kullback-Leibler divergence equation (KL divergence)
#The result of this equation is added to the loss function result as an additional penalty to the loss based on sparsity
def KL_Div(rho, rho_hat):

    KL_loss = rho * logfunc(rho, rho_hat) + (1 - rho) * logfunc((1 - rho), (1 - rho_hat))
    
    #rescaling KL result to 0-1 range
    return tf.div(KL_loss-tf.reduce_min(KL_loss)+1e-10,tf.reduce_max(KL_loss)-tf.reduce_min(KL_loss))
    #RR I just simplified the KL divergence equation according to the book:
    #RR "Hands-On Machine Learning with Scikit-Learn and TensorFlow" by Aurélien Géron
    #RR Example source code here https://github.com/zhiweiuu/sparse-autoencoder-tensorflow/blob/master/SparseAutoEncoder.py
    #RR KL2 is the classic sparsity implementation, source reference: https://github.com/elykcoldster/sparse_autoencoder/blob/master/mnist_sae.py
    #https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf


#drop out function will exclude samples from our input data
#keep_rate will determine how many samples will remain after randomly droping out samples from the input
def dropout(input, name, keep_rate):
    with tf.name_scope(name):
        out = tf.nn.dropout(input, keep_rate)
    return out
    # call function like this, p1 is input, name is layer name, and keep rate doesnt need explanation,  
    # do1 = dropout(p1, name='do1', keep_rate=0.75)

    #A value of 1.0 means that dropout will not be used.
    #TensorFlow documentation https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#dropout


# In[ ]:


#Example adapted and modified from 
#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

#Encodes a Hidden layer
def encoder(x, func, l1_val, l2_val, weights, biases, units_num, keep_rate=1): #RR added keep_rate
    print("Setting up encoder/decoder.")
    regularizer = tf.contrib.layers.l1_l2_regularizer(l1_val,l2_val)

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
    elif func == 'relu':
        print('Encoder Activation function: relu')
        #with tf.device("/gpu:1"):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
        layer_1 = tf.layers.dense(layer_1, units=units_num, kernel_regularizer= regularizer)
        #layer_1 = tf.layers.dense(layer_1, units=221, kernel_regularizer= regularizer)

    return layer_1
        
def decoder(x, func, weights, biases):
    if func == 'sigmoid':
        print('Decoder Activation function: sigmoid')
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        #rescaling, if dealing with categorical	variables or factors, tf.reduce_max(x) will result in 1
        layer_1 = tf.multiply(layer_1, tf.reduce_max(x), name="y_pred")
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    elif func == 'tanh':
        print('Decoder Activation function: tanh')
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        #rescaling, if dealing with categorical variables or factors, tf.reduce_max(x) will result in 1
        layer_1 = tf.div(tf.multiply(tf.add(layer_1, 1), tf.reduce_max(x)), 2, name="y_pred")
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    elif func == 'relu':
        print('Decoder Activation function: relu')
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']), name="y_pred")
        #no rescaling needed
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    return layer_1


# In[ ]:
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

        return r2hat[0]
     
    j=0
    mean_r2hat = 0
    r2hats = []
    MAFs = calculate_MAF(x, categorical)
    idx = 0
    
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
        
        r2hat = calculate_r2hat(x_genotypes, MAFs[idx])
        r2hats.append(r2hat)
        if(r2hat>0):
            mean_r2hat += r2hat/len(MAFs)
        idx += 1 

    return mean_r2hat, r2hats

def accuracy_maf_threshold(sess, x, y, threshold1, threshold2, categorical=False):
    
    indexes_to_keep = []
    
    MAFs = calculate_MAF(y, categorical)
    
    i = 0
    j = 0
    k = 0
    
    while i < len(MAFs):
        if(MAFs[i]>=threshold1 and MAFs[i]<=threshold2):
            if(categorical==True):
                indexes_to_keep.append(j)
                indexes_to_keep.append(j+1)
                indexes_to_keep.append(j+2)
            elif(categorical==False):
                indexes_to_keep.append(k)
                indexes_to_keep.append(k+1)            
        i += 1
        j += 3
        k += 2
        
    #print(indexes_to_keep)
    #print(len(x[0]))
    #print(len(y[0]))
    getter = operator.itemgetter(indexes_to_keep)
    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
    #print(len(filtered_data_x))
    #print(len(filtered_data_y[0]))

    correct_prediction = sess.run(tf.equal( tf.round( filtered_data_x ), tf.round( filtered_data_y ) ))
    accuracy = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    accuracy_per_marker = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 0))

    return accuracy, accuracy_per_marker

def MSE_maf_threshold(sess, x, y, threshold1, threshold2, categorical=False):
    
    indexes_to_keep = []
    
    MAFs = calculate_MAF(y, categorical)
    
    i = 0
    j = 0
    k = 0
    
    while i < len(MAFs):
        if(MAFs[i]>=threshold1 and MAFs[i]<=threshold2):
            if(categorical==True):
                indexes_to_keep.append(j)
                indexes_to_keep.append(j+1)
                indexes_to_keep.append(j+2)
            elif(categorical==False):
                indexes_to_keep.append(k)
                indexes_to_keep.append(k+1)            
        i += 1
        j += 3
        k += 2
        
    #print(indexes_to_keep)
    #print(len(x[0]))
    #print(len(y[0]))
    getter = operator.itemgetter(indexes_to_keep)
    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
    #print(len(filtered_data_x))
    #print(len(filtered_data_y[0]))

    MSE = sess.run( tf.reduce_mean(tf.square( tf.subtract( tf.round( filtered_data_x ), tf.round( filtered_data_y ) ) ) ) )
    MSE_per_marker = sess.run( tf.reduce_mean(tf.square( tf.subtract( tf.round( filtered_data_x ), tf.round( filtered_data_y ) ) ) ), 0 )

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

#Code modified from example
#https://stackoverflow.com/questions/44367010/python-tensor-flow-relu-not-learning-in-autoencoder-task
def run_autoencoder(learning_rate, training_epochs, l1_val, l2_val, act_val, beta, rho, keep_rate, data_obs):

    print("Running autoencoder.")
    # parameters
    #learning_rate = 0.01
    #training_epochs = 50
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
    if(len(data_obs.shape) == 3):
        n_input = len(data_obs[0])*len(data_obs[0][0])
    else:
        n_input = len(data_obs[0])*3     # input features N_variants*subfeatures
        
    n_hidden_1 = n_input  # hidden layer for encoder, equal to input number of features for now
    print("Input data shape after coding variables:")
    print(n_input)
    
    if(recovery_mode==False):
    #tf input
        X = tf.placeholder("float", [None, n_input], name="X")
        #print(X)
        Y = tf.placeholder("float", [None, n_input], name="Y")
    else:
        X = tf.placeholder("float", [None, n_input], name="newX")
    #    #print(X)
        Y = tf.placeholder("float", [None, n_input], name="newY")
    #As parameters of a statistical model, weights and biases are learned or estimated by minimizing a loss function that depends on our data. 
    #We will initialize them here, their values will be set during the learning process
    #def define_weights_biases(n_input, n_hidden_1):
    #with tf.device("/gpu:1"):
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="w_encoder_h1"),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input]), name="w_decoder_h1"),
    }
    
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1]), name="b_encoder_b1"),
        'decoder_b1': tf.Variable(tf.random_normal([n_input]), name="b_decoder_b1"),
    }
    #print(X.get_shape())
    keep_prob = tf.placeholder("float", None, name="keep_prob") ## RR adding dropout
    
    # construct model
    encoder_op = encoder(X, act_val, l1_val, l2_val, weights, biases, n_input, keep_rate)
    encoder_result = encoder_op
    decoder_op = decoder(encoder_op, act_val, weights, biases)

    #example from test prediction, take this out from the training
    #test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    
    #print(encoder_op)
    # predict result
    y_pred = decoder_op
    # real input data as labels
    y_true = Y
    
    rho_hat = tf.reduce_mean(encoder_op,0) #RR sometimes returns Inf in KL function, caused by division by zero, fixed wih logfun()
    if (act_val == "tanh"):
        rho_hat = tf.div(tf.add(rho_hat,1.0),2.0) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl
    if (act_val == "relu"):
        rho_hat = tf.div(tf.add(rho_hat,1e-10),tf.reduce_max(rho_hat)) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl

    rho = tf.constant(rho) #not necessary maybe?
        
    #sparsity_loss =  tf.clip_by_value(tf.reduce_sum(KL_Div(rho, rho_hat)), 1e-10, 1.0) #RR KL divergence, clip to avoid Inf or div by zero
    sparsity_loss =  tf.reduce_mean(KL_Div(rho, rho_hat), name="sparsity_loss")
    # define cost function, optimizers
    # loss function: MSE # example cost = tf.reduce_mean(tf.square(tf.subtract(output, x)))
    reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred)), name="reconstruction_loss") #RR simplified the code bellow
    # newly added sparsity function (RHO, BETA)
    #cost_sparse = tf.multiply(beta,  tf.reduce_sum(KL_Div(RHO, rho_hat)))
    #print(cost_sparse)
    cost = tf.reduce_mean(reconstruction_loss + beta * sparsity_loss, name = "cost") #RR simplified equation
        
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
    
    if(factors == False): #only factors==False is working now, TODO: fix the else: bellow
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

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
    cost_accuracy = tf.add((1-accuracy), beta * sparsity_loss, name="cost_accuracy")

    #The RMSprop optimizer is similar to the gradient descent algorithm with momentum. 
    #The RMSprop optimizer restricts the oscillations in the vertical direction. 
    #Therefore, we can increase our learning rate and our algorithm could take larger steps in the horizontal direction converging faster. 
    #ref: https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b
    if(recovery_mode==False):
        #Converge by accuracy when using factors
        if(factors == False):
            optimizer = tf.train.RMSPropOptimizer(learning_rate, name="optimizer").minimize(cost)
        else:
            optimizer = tf.train.RMSPropOptimizer(learning_rate, name="optimizer").minimize(cost_accuracy)
        #save this optimizer to restore it later.
        #tf.add_to_collection("optimizer", optimizer)
        #tf.add_to_collection("Y", Y)
        #tf.add_to_collection("X", X)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # initialize variables
    if(save_model==True):
        #Create a saver object which will save all the variables
        saver = tf.train.Saver(tf.all_variables(),max_to_keep=10)
        
    #init = tf.global_variables_initializer();
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    start = timeit.default_timer()
    # run autoencoder .........
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    config = tf.ConfigProto(log_device_placement=False)
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4
    config.gpu_options.per_process_gpu_memory_fraction = 0.20
    config.gpu_options.allow_growth=True
    
    with tf.device('/device:GPU:0'):  # Replace with device you are interested in
        bytes_in_use = BytesInUse()
    
    if(recovery_mode==True):
        tf.reset_default_graph()
    
    with tf.Session(config=config) as sess:        
        
        if(recovery_mode==True):
            #recover model from checkpoint
            meta_path = model_path + '.meta'    
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, model_path)
            graph = sess.graph
            #optimizer = graph.get_tensor_by_name("optimizer:0")
            optimizer = graph.get_operation_by_name( "optimizer" )
            X = "X:0"
            Y = "Y:0"
            cost = "cost:0"
            reconstruction_loss = "reconstruction_loss:0"
            sparsity_loss = "sparsity_loss:0"            
            accuracy = "accuracy:0"
            cost_accuracy = "cost_accuracy:0"
            y_pred = "y_pred:0"
        else:    
            sess.run(init)
        
        mean_cost = 0
        mean_rloss = 0
        mean_sloss = 0
        mean_acc = 0
        mean_cacc = 0    
        ki = 0
        mean_acc_t = [0,0,0,0,0,0,0]
        my_acc_t = [0,0,0,0,0,0,0]

        data_idx = 0        

        if(kn>=2):
            kf = KFold(n_splits=kn)
        else:
            kf = KFold(n_splits=10)    

        data_masked = np.zeros(data_obs.shape)
        #print(data_obs.shape)
        data_idx = kf.split(data_masked, data_obs)

        if(fixed_masking == True): #mask all data only once before the whole training procedure
            if(mask_per_sample == False):
                data_masked, _ = mask_data(np.copy(data_obs),0.9,sys.argv[len(sys.argv)-1])
            else:
                data_masked = mask_data_per_sample(np.copy(data_obs), 0.9, sys.argv[len(sys.argv)-1])
                       
            data_masked = flatten_data(sess, np.copy(data_masked), factors)
            data_obs = flatten_data(sess, np.copy(data_obs), factors)
            
            print(data_masked.shape)

        for train_idx, val_idx in data_idx:

            if(kn>=2):
                if(fixed_masking == True):
                    train_x = data_masked[train_idx]
                    val_x = data_masked[val_idx]
                else: #if fixed masking is false mask later at the begining of each epoch
                    train_x = data_obs[train_idx]
                    val_x = data_obs[val_idx]
                    
                train_y = data_obs[train_idx]
                val_y = data_obs[val_idx]
            else:
                if(fixed_masking == True):
                    train_x = np.copy(data_masked)
                    del data_masked
                else:
                    train_x = np.copy(data_obs)                    
                train_y = np.copy(data_obs)

            total_batch = int(train_x.shape[0] / batch_size)
            print(train_x.shape)
            
            ki += 1
            #create a backup of the original training data
            if(initial_masking_rate > 0 and fixed_masking == False):
                mask_rate = initial_masking_rate
            else:
                mask_rate = 0.9 #set default 
            
            for epoch in range(training_epochs+1):
                    
                if(fixed_masking == False):
                    #make new masking on every new iteration
                    if(recovery_mode==False and epoch!=0):
                        if(mask_per_sample == True):
                            #print(train_x.shape)
                            data_masked = mask_data_per_sample(np.copy(data_obs),mask_rate,sys.argv[len(sys.argv)-1])
                        else:
                            data_masked, _ = mask_data(np.copy(data_obs),mask_rate,sys.argv[len(sys.argv)-1])

                        if(kn>=2):
                            train_y = np.copy(data_obs[train_idx])
                            train_x = data_masked[train_idx]  
                            val_y = np.copy(data_obs[val_idx])
                            val_x = data_masked[val_idx]    
                            val_x = flatten_data(sess, np.copy(val_x), factors)
                            val_y = flatten_data(sess, np.copy(val_y), factors)                                
                        else:
                            
                            train_x = data_masked
                            train_y = np.copy(data_obs)
                            
                    #after masking, flatten data
                    train_x = flatten_data(sess, np.copy(train_x), factors)
                    train_y = flatten_data(sess, np.copy(train_y), factors)
                    
                    mask_rate += initial_masking_rate
                    
                for i in range(total_batch):
                    batch_x = train_x[i*batch_size:(i+1)*batch_size]
                    batch_y = train_y[i*batch_size:(i+1)*batch_size]
                    
                    #calculate cost and optimizer functions
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y} )
                                                
                if(save_pred==True and ki==1): #if using cluster run for all k fold: if(save_pred==True):
                    #This generates 1GB files per epoch per k-fold iteration, at least 5TB of space required, uncoment when using the HPC cluster with big storage
                    #my_pred = sess.run([y_pred], feed_dict={X: train_x, Y: train_y} )
                    #print(my_pred.shape) #3D array 1,features,samples
                    #fname = "k-" + str(ki) + "_epoch-" + str(epoch+1) + "_train-pred.out"
                    #with open(fname, 'w') as f:
                    #    np.savetxt(f, my_pred[0])
                    #f.close()

                    my_pred = sess.run([y_pred], feed_dict={X: val_x, Y: val_y} )
                    
                    fname = "k-" + str(ki) + "_epoch-" + str(epoch+1) + "_val-pred.out"
                    with open(fname, 'w') as f:
                        np.savetxt(f,  my_pred[0])
                    f.close()


            if(save_pred==True and kn>=2): #if kn<2, validation and training are the same
                fname = "k-" + str(ki) + "_val-obs.out"
                with open(fname, 'w') as f:
                    np.savetxt(f, val_y)
                f.close()

                fname = "k-" + str(ki) + "_val-input.out"
                with open(fname, 'w') as f:
                    np.savetxt(f, val_x)
                f.close()

            if(save_pred==True):

                fname = "k-" + str(ki) + "_train-obs.out"
                with open(fname, 'w') as f:
                    np.savetxt(f, train_y)
                f.close()

                fname = "k-" + str(ki) + "_train-input.out"
                with open(fname, 'w') as f:
                    np.savetxt(f, train_x)
                f.close()
                
            if(kn>=2):
                my_cost, my_rloss, my_sloss, my_acc, my_cacc  = sess.run([cost, reconstruction_loss, sparsity_loss, accuracy, cost_accuracy], feed_dict={X: val_x, Y: val_y})
                my_pred = sess.run([y_pred], feed_dict={X: val_x, Y: val_y})
                my_pred = my_pred[0]

                my_acc_t[0], acc_per_m = accuracy_maf_threshold(sess, my_pred, val_y, 0, 1, factors)
                my_acc_t[1], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0, 0.005, factors)
                my_acc_t[2], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0.005, 1, factors)
                my_acc_t[3], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0, 0.001, factors)
                my_acc_t[4], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0.001, 1, factors)
                my_acc_t[5], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0, 0.01, factors)
                my_acc_t[6], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0.01, 1, factors)
                
                my_MSE, MSE_list = accuracy_maf_threshold(sess, my_pred, val_y, 0, 1, factors)

                my_MAF_list = calculate_MAF(val_y, factors)
                mean_r2hat_est, my_r2hat_est = mean_estimated_r2(val_y, factors)
                mean_r2hat_emp, my_r2hat_emp = mean_empirical_r2(val_x, val_y, factors)


            else:
                my_cost, my_rloss, my_sloss, my_acc, my_cacc  = sess.run([cost, reconstruction_loss, sparsity_loss, accuracy, cost_accuracy], feed_dict={X: train_x, Y: train_y})
                
                #my_pred = sess.run([y_pred], feed_dict={X: train_x, Y: train_y})
                my_pred = sess.run([y_pred], feed_dict={X: train_x, Y: train_y})
                my_pred = my_pred[0]
                print("my_pred")
                print(len(my_pred))
                print(len(my_pred[0]))
                print("my_train")
                print(len(train_x))
                print(len(train_x[0]))
                
                my_acc_t[0], acc_per_m = accuracy_maf_threshold(sess, my_pred, train_y, 0, 1, factors)
                my_acc_t[1], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0, 0.005, factors)
                my_acc_t[2], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0.005, 1, factors)
                my_acc_t[3], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0, 0.001, factors)
                my_acc_t[4], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0.001, 1, factors)
                my_acc_t[5], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0, 0.01, factors)
                my_acc_t[6], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0.01, 1, factors)

                my_MSE, MSE_list = accuracy_maf_threshold(sess, my_pred, train_y, 0, 1, factors)

                my_MAF_list = calculate_MAF(train_y, factors)
                mean_r2hat_est, my_r2hat_est = mean_estimated_r2(train_y, factors)
                mean_r2hat_emp, my_r2hat_emp = mean_empirical_r2(train_x, train_y, factors)
                
            if(save_model==True):
                #Now, save the graph
                filename='./inference_model-' + str(ki) + ".ckpt"
                print("Saving model to file:", filename)
                saver.save(sess, filename)
                
            print("Maximum VRAM used: ")
            # maximum across all sessions and .run calls so far
            print(sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
            #print(sess.run(bytes_in_use))

            #print("MAF", len(my_MAF_list), "acc", len(acc_per_m), "r2emp", len(my_r2hat_emp), "r2est", len(my_r2hat_emp), "MSE", len(MSE_list))
            
            idx=1
            for i in range(len(my_MAF_list)):
                print("METRIC_MAF:", my_MAF_list[i])
                print("METRIC_acc_per_m:", acc_per_m[idx])
                print("METRIC_r2_emp:", my_r2hat_est[i])
                print("METRIC_r2_est:", my_r2hat_emp[i])
                print("METRIC_MSE_per_m:", MSE_list[idx])
                idx+=2

                
            mean_cost += my_cost/kn
            mean_rloss += my_rloss/kn
            mean_sloss += my_sloss/kn
            mean_acc += my_acc/kn
            mean_cacc += my_cacc/kn
            
            for j in range(len(mean_acc_t)):
                mean_acc_t[j] += my_acc_t[j]/kn
            
            
            if(kn<=1):
                print("Training done, not running CV.")
                break
            else:            
                print("K-fold iteration: ", ki, " of ", split_size, ".")
        
        print("Mean accuracy for MAF 0-1 range:", mean_acc_t[0])
        print("Mean accuracy for MAF 0-0.005 range:", mean_acc_t[1])
        print("Mean accuracy for MAF 0.005-1 range:", mean_acc_t[2])
        print("Mean accuracy for MAF 0-0.001 range:", mean_acc_t[3])
        print("Mean accuracy for MAF 0.001-1 range:", mean_acc_t[4])
        print("Mean accuracy for MAF 0-0.01 range:", mean_acc_t[5])
        print("Mean accuracy for MAF 0.01-1 range:", mean_acc_t[6])
        print("MSE MAF 0-1 range:", my_MSE)
        print("r2hat est. MAF 0-1 range:", mean_r2hat_est)
        print("r2hat emp. MAF 0-1 range:", mean_r2hat_emp)


        #pbar.close()
        print("Optimization finished!")
        
    stop = timeit.default_timer()
    print('Time to run all training (sec): ', stop - start)
    print('Time to run each k-fold step (sec): ', (stop - start)/kn)
    print('Time to run each epoch, average (sec): ', (stop - start)/training_epochs*total_batch)
    #reset tensorflow session
    tf.reset_default_graph()
    sess.close()
    # return the minimum loss of this combination of L1, L2, activation function, beta, rho on a dataset
    #return minLoss, min_reconstruction_loss, min_sparsity_loss, max_correl
    return mean_cost, my_rloss, mean_sloss, mean_acc, mean_cacc


# In[ ]:


def main():
    keep_rate = 1
    split_size = 10 #k for 10-fold cross-validation
    
    print("This is the name of the script: ", sys.argv[0])
    print("Number of arguments: ", len(sys.argv))
    print("The arguments are: " , str(sys.argv))
    mask_rate=0.9
    
    kf = KFold(n_splits=split_size)      

    if(len(sys.argv)==4):
        print("Parsing input file: ")
        #Arguments
        #sys.argv[1] = [str] input file (HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf)
        #sys.argv[2] = [str] hyperparameter list file (mylist.txt)
        #sys.argv[3] = [True,False] Use categorical coding for variables instead of counts, default is False

        data_obs = process_data(sys.argv[1],sys.argv[3]) #input file, i.e: HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf
        #data_masked = mask_data(np.copy(data_obs),mask_rate,sys.argv[3])

        print("reading hyperparameter list from file: ", sys.argv[2])
        hp_array = []
        result_list = []

        with open(sys.argv[2]) as my_file:
            for line in my_file:                
                hp_array.append(line.split())
        i = 0
        while(i < len(hp_array)):
            
            l1 = float(hp_array[i][0])
            l2 = float(hp_array[i][1])
            beta = float(hp_array[i][2])
            rho = float(hp_array[i][3])
            act = str(hp_array[i][4])
            lr = float(hp_array[i][5])
            tmp_list = hp_array[i]
            print("Starting CV... Parameters:", tmp_list)
            my_cost, my_rloss, my_sloss, my_acc, my_cacc = run_autoencoder(lr, training_epochs, l1, l2, act, beta, rho, keep_rate, data_obs)
            tmpResult = [my_cost, my_rloss, my_sloss, my_acc, my_cacc]
            tmp_list.extend(tmpResult)
            result_list.append(tmp_list)
            i += 1
            print("TMP_RESULT: ", tmp_list)
        return result_list
            
    else:   
        data_obs = process_data(sys.argv[1],sys.argv[8]) #input file, i.e: HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf
        #data_masked = mask_data(np.copy(data_obs),0.9,sys.argv[8])
        #Arguments
        #sys.argv[1] = [str] input file (HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf)
        #sys.argv[2] = [float] L1 hyperparameter value
        #sys.argv[3] = [float] L2 hyperparameter value
        #sys.argv[4] = [float] Sparsity beta hyperparameter value
        #sys.argv[5] = [float] Sparseness rho hyperparameter value
        #sys.argv[6] = [str] Activation function ('anh')
        #sys.argv[7] = [float] Learning rate hyperparameter value
        #sys.argv[8] = [True,False] Use categorical coding for variables instead of counts, default is False
        l1 = float(sys.argv[2])
        l2 = float(sys.argv[3])
        beta = float(sys.argv[4])
        rho = float(sys.argv[5])
        act = str(sys.argv[6])
        lr = float(sys.argv[7])
        #grid = make_grid(l1_arr, l2_arr, beta_arr, rho_arr, act_arr,learning_rate_arr,N)
        tmp_list = sys.argv[2:]

        my_cost, my_rloss, my_sloss, my_acc, my_cacc = run_autoencoder(lr, training_epochs, l1, l2, act, beta, rho, keep_rate, data_obs)

        tmpResult = [my_cost, my_rloss, my_sloss, my_acc, my_cacc]
        #tmp_list.extend(tmpResult)
        #result_list.append(tmp_list)
        #i += 1
        #tmpResult = run_autoencoder(lr, training_epochs, l1, l2,act, beta, rho, masked, keep_rate, curr_data)
        # append the average loss of this combination of L1, L2, activation function, beta, rho
        tmp_list.extend(tmpResult)
        return tmp_list



if __name__ == "__main__":
    result = main()
    print("LABELS [L1, L2, BETA, RHO, ACT, LR, MSE+BETA*SPARSE_COST, MSE, SPARSE_COST, ACCURACY, (1-ACCURACY)+BETA*SPARSE_COST]") 
    if(len(sys.argv)==4):
        i = 0
        while(i < len(result)):
            print("RESULT ", result[i])
            i += 1
    else:
        print("RESULT ", result)

