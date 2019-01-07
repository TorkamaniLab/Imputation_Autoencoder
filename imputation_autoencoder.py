
# coding: utf-8

# In[ ]:


#example
#/bin/python3.6 imputation_autoencoder.py imputed_test_subset.vcf 3_hyper_par_set.txt True
#CUDA_VISIBLE_DEVICES=0 /bin/python3.6 imputation_autoencoder.py imputed_test_subset.vcf 3_hyper_par_set.txt True
#CUDA_VISIBLE_DEVICES=1 /bin/python3.6 imputation_autoencoder.py imputed_test_subset.vcf 0.1 0.01 2 0.7 relu 0.001 False

import tensorflow as tf
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

from scipy.stats import pearsonr

from tensorflow.python.client import device_lib #get_available_devices()

from joblib import Parallel, delayed #parallel tasks
import multiprocessing #parallel tasks

from sklearn.model_selection import KFold

import sys #arguments

###################################OPTIONS#############################################
my_keep_rate = 1 #keep rate for dropout funtion, 1=disable dropout

save_pred = False #[True,False] save predicted results for each training epoch and for k-fold CV

split_size = 10 #number of batches

kn = 1 #number of k for k-fold CV (kn>=2); if k=1: just run training

training_epochs = 500 #default 500

###################################OPTIONS#############################################
    
# In[ ]:

# parsing vcf files, 
#If categorical false: convert 1|1 -> [0,2], 1|0 or 0|1 -> [1,1], 0|0 -> [2,0], missing -> [0,0]
#If categorical true: convert 1|1 -> 2, 1|0 or 0|1 -> 1, 0|0 -> 0, missing -> 4

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
    new_df = np.zeros((len(df_T)-9,len(df),2))
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
    # random matrix the same shape of your data
    #print(len(mydata))
    nmask = int(round(len(mydata[0])*mask_rate))
    # random boolean mask for which values will be changed
    maskindex = random.sample(range(0, len(mydata[0]-1)), nmask)
    #np.random.randint(0,len(mydata[0]),size=nmask)
    print("Masking markers...")
    print(maskindex)
    #mydata = np.transpose(mydata)
    print(mydata.shape)
    #mydata
    
    #pbar = tqdm(total = len(maskindex))
    #for i in range(10):
    #    print(mydata[i][0:11])


    for i in maskindex:
        #print(len(mydata[i]))
        j = 0
        while j < len(mydata):
            if(categorical=="True"):
                mydata[j][i]=-1
            else:
                mydata[j][i]=[0,0]
            j=j+1
        #pbar.update(1)
        #print(mydata[i])
    #mydata = np.transpose(mydata)
    #print(mydata.shape)
    #pbar.close()

    #print("after masking:")
    #for i in range(10):
    #    print(mydata[i][0:11])

    stop = timeit.default_timer()
    print('Time to mask the data (sec): ', stop - start)  
    return mydata


# In[ ]:


def logfunc(x, x2):
    cx = tf.clip_by_value(x, 1e-10, 1.0-1e-10)
    cx2 = tf.clip_by_value(x2, 1e-10, 1.0-1e-10)
    return tf.multiply( x, tf.log(tf.div(cx,cx2)))


#Kullback-Leibler divergence equation (KL divergence)
#The result of this equation is added to the loss function result as an additional penalty to the loss based on sparsity
def KL_Div(rho, rho_hat):
    #rho = tf.clip_by_value(rho, 1e-10, 1.0-1e-10) # prevent nan on log(0)
    #rho_hat = tf.clip_by_value(rho_hat, 1e-10, 1.0-1e-10) # prevent nan on log(0)
    #rho = tf.add(rho, tf.constant(1e-10)) #prevent nan on log(0)
    #rho_hat = tf.add(rho_hat, tf.constant(1e-10)) #prevent nan on log(0)
    #KL_loss = rho * tf.log(tf.clip_by_value(rho / rho_hat, 1e-10, 1.0-1e-10)) + (1 - rho) * tf.log(tf.clip_by_value((1 - rho) / (1 - rho_hat), 1e-10, 1.0-1e-10))
    #KL_loss = rho * tf.log(rho / rho_hat) + (1 - rho) * tf.log((1 - rho) / (1 - rho_hat))

    KL_loss = rho * logfunc(rho, rho_hat) + (1 - rho) * logfunc((1 - rho), (1 - rho_hat))
    return tf.clip_by_value(KL_loss, 1e-10, 1.0-1e-10)
    
    #invrho = tf.subtract(tf.constant(1.), rho)
    #invrhohat = tf.subtract(tf.constant(1.), rho_hat)
    #KL_loss2 = tf.add(logfunc(rho,rho_hat), logfunc(invrho, invrhohat))
    #return tf.clip_by_value(KL_loss2, 1e-10, 1.0-1e-10)
    
    #return rho * tf.log(rho / rho_hat) + (1 - rho) * tf.log((1 - rho) / (1 - rho_hat))
    #return tf.reduce_sum(tf.multiply(rho, tf.log(tf.div(rho,rho_hat)),tf.multiply(tf.subtract(1,rho),tf.log(tf.div(tf.subtract(1,rho),tf.subtract(1,rho_hat))))))
    #RR I just simplified the KL divergence equation according to the book:
    #RR "Hands-On Machine Learning with Scikit-Learn and TensorFlow" by Aurélien Géron
    #RR Example source code here https://github.com/zhiweiuu/sparse-autoencoder-tensorflow/blob/master/SparseAutoEncoder.py
    #RR KL2 is the classic sparsity implementation, source reference: https://github.com/elykcoldster/sparse_autoencoder/blob/master/mnist_sae.py
    #https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf


# In[ ]:


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
        layer_1 = tf.multiply(layer_1, tf.reduce_max(x))
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    elif func == 'tanh':
        print('Decoder Activation function: tanh')
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        #rescaling, if dealing with categorical variables or factors, tf.reduce_max(x) will result in 1
        layer_1 = tf.div(tf.multiply(tf.add(layer_1, 1), tf.reduce_max(x)), 2)
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    elif func == 'relu':
        print('Decoder Activation function: relu')
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        #no rescaling needed
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    return layer_1


# In[ ]:


def pearson_correl(x,y):
    x = np.transpose(np.copy(x))
    y = np.transpose(np.copy(y))
    cor_result = 0
    #print(len(x))
    #print(x[185])
    nans = 0
    for i in range(len(x)):
        cor,p = pearsonr(x[i],y[i])
        #print(i)
        if(np.isnan(cor)):
            nans += 1
        else:
            cor_result += cor
            #print(cor)
    if (len(x)-nans) == 0 or cor_result == 0:
        return 0
    else:
        return cor_result/(len(x)-nans)


# In[ ]:


#Code modified from example
#https://stackoverflow.com/questions/44367010/python-tensor-flow-relu-not-learning-in-autoencoder-task
def run_autoencoder(learning_rate, training_epochs, l1_val, l2_val, act_val, beta, rho, data_masked, keep_rate, data_obs):

    print("Running autoencoder.")
    # parameters
    #learning_rate = 0.01
    #training_epochs = 50
    factors = True
    #subjects, SNP, REF/ALT counts
    if(len(data_masked.shape) == 3):
        #data_masked = np.reshape(data_masked, (data_masked.shape[0],data_masked.shape[1]*2))
        #data_obs = np.reshape(data_obs, (data_obs.shape[0],data_obs.shape[1]*2))
        data_masked = np.reshape(data_masked, (data_masked.shape[0],-1))
        data_obs = np.reshape(data_obs, (data_obs.shape[0],-1))
        factors = False
    else:#do one hot encoding, depth=3 because missing (-1) is encoded to all zeroes
        data_masked =  tf.one_hot(indices=data_masked, depth=3)
        data_obs =  tf.one_hot(indices=data_obs, depth=3)
        data_masked = tf.layers.flatten(data_masked)#flattening to simplify calculations later (matmul, add, etc)
        data_obs = tf.layers.flatten(data_obs)

    print("Input data shape:")
    print(data_masked.shape)
    print(data_obs.shape)

    batch_size = int(round(len(data_masked)/split_size)) # size of training objects split the dataset in 10 parts
    #print(batch_size)
    
    display_step = 1        

    # define layer size
    n_input = len(data_masked[0])     # input features N_variants
    n_hidden_1 = n_input  # hidden layer for encoder, equal to input number of features for now
    
    
    #tf input
    X = tf.placeholder("float", [None, n_input])
    #print(X)
    Y = tf.placeholder("float", [None, n_input])

    #As parameters of a statistical model, weights and biases are learned or estimated by minimizing a loss function that depends on our data. 
    #We will initialize them here, their values will be set during the learning process
    #def define_weights_biases(n_input, n_hidden_1):
    #with tf.device("/gpu:1"):
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }
    
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b1': tf.Variable(tf.random_normal([n_input])),
    }
    #print(X.get_shape())
    keep_prob = tf.placeholder("float", None) ## RR adding dropout
    
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
    
    # sparsity 
    #beta = 0.001 #RR beta is a search grid parameter, not hardcoded anymore
    #act_sum = tf.reduce_sum(encoder_op,0)  
    #N = tf.constant(float(batch_size))
    #rho_hat = tf.div(act_sum,N)
    rho_hat = tf.reduce_mean(encoder_op,0) #RR sometimes returns Inf in KL function, caused by division by zero, fixed wih logfun()
    if (act_val == "tanh"):
        rho_hat = tf.div(tf.add(rho_hat,1.0),2.0) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl
    if (act_val == "relu"):
        rho_hat = tf.div(tf.add(rho_hat,1e-10),tf.reduce_max(rho_hat)) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl

    rho = tf.constant(rho) #not necessary maybe?
        
    sparsity_loss =  tf.clip_by_value(tf.reduce_sum(KL_Div(rho, rho_hat)), 1e-10, 1.0) #RR KL divergence, clip to avoid Inf or div by zero
    # define cost function, optimizers
    # loss function: MSE # example cost = tf.reduce_mean(tf.square(tf.subtract(output, x)))
    reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred))) #RR simplified the code bellow
    # newly added sparsity function (RHO, BETA)
    #cost_sparse = tf.multiply(beta,  tf.reduce_sum(KL_Div(RHO, rho_hat)))
    #print(cost_sparse)
    cost = tf.reduce_mean(reconstruction_loss + beta * sparsity_loss) #RR simplified equation
        
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
    
    if(factors == False):
        correct_prediction = tf.equal( tf.round( y_pred ), tf.round( y_true ) )
    else:
        y_pred_shape = tf.shape(y_pred)
        y_true_shape = tf.shape(y_true)
        #reshaping back to original form, so the argmax function can work
        y_pred = tf.reshape(y_pred, [y_pred_shape[0], y_pred_shape[1]/3, 3] ) 
        y_true = tf.reshape(y_true, [y_pred_shape[0], y_pred_shape[1]/3, 3] )
        y_pred = tf.argmax( y_pred, 1 ) #back to original categorical form [-1,0,1,2]
        y_true = tf.argmax( y_true, 1 ) #back to original categorical form [-1,0,1,2]
        correct_prediction = tf.equal( y_pred, y_true )

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cost_accuracy = (1-accuracy) + beta * sparsity_loss

    #The RMSprop optimizer is similar to the gradient descent algorithm with momentum. 
    #The RMSprop optimizer restricts the oscillations in the vertical direction. 
    #Therefore, we can increase our learning rate and our algorithm could take larger steps in the horizontal direction converging faster. 
    #ref: https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b
    
    #Converge by accuracy when using factors
    if(factors == False):
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_accuracy)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    #print("sparsity_loss")
    #print(sparsity_loss)
    #print("reconstruction_loss")
    #print(reconstruction_loss)
    #print("cost")
    #print(cost)
    # initialize variables
    
    #init = tf.global_variables_initializer();
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # store the result epochs and loss 
    #tmpResult = []
    #tmpResult2 = []
    #tmpResult3 = []
    #tmpResult4 = []
    #tmpResult5 = []
    start = timeit.default_timer()
    # run autoencoder .........
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    config = tf.ConfigProto(log_device_placement=False)
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4
    config.gpu_options.per_process_gpu_memory_fraction = 0.15
    
    with tf.Session(config=config) as sess:        
        sess.run(init)

        mean_cost = 0
        mean_rloss = 0
        mean_sloss = 0
        mean_acc = 0
        mean_cacc = 0    
        ki = 0

        data_idx = 0        

        if(kn>=2):
            kf = KFold(n_splits=kn)
        else:
            kf = KFold(n_splits=10)
            
        data_idx = kf.split(data_masked, data_obs)

        for train_idx, val_idx in data_idx:

            if(kn>=2):
                train_x = data_masked[train_idx]
                train_y = data_obs[train_idx]
                val_x = data_masked[val_idx]
                val_y = data_obs[val_idx]
            else:
                train_x = np.copy(data_masked)
                del data_masked
                train_y = np.copy(data_obs)
                del data_obs

            total_batch = int(train_x.shape[0] / batch_size)

            ki += 1

            for epoch in range(training_epochs):
                
                for i in range(total_batch):
                    batch_x = train_x[i*batch_size:(i+1)*batch_size]
                    batch_y = train_y[i*batch_size:(i+1)*batch_size]
                   
                    # calculate cost and optimizer functions
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y} )

                if(save_pred==True and ki==1): #if using cluster run for all k fold: if(save_pred==True):
                    #This generates 1GB files per epoch per k-fold iteration, at least 5TB of space required, uncoment when using the HPC cluster
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
            else:
                my_cost, my_rloss, my_sloss, my_acc, my_cacc  = sess.run([cost, reconstruction_loss, sparsity_loss, accuracy, cost_accuracy], feed_dict={X: train_x, Y: train_y})

            mean_cost += my_cost/kn
            mean_rloss += my_rloss/kn
            mean_sloss += my_sloss/kn
            mean_acc += my_acc/kn
            mean_cacc += my_cacc/kn

            if(kn<=1):
                print("Training done, not running CV.")
                break
            else:            
                print("K-fold iteration: ", ki, " of ", split_size, ".")

        #tmpResult.append(mean_cost) ##RR adding average cost
        #tmpResult2.append(mean_rloss)
        #tmpResult3.append(mean_sloss)
        #tmpResult4.append(mean_acc)
        #tmpResult5.append(mean_cacc)
        #pbar2.close()
        #pbar.update(1)
        #print('Epoch', epoch+1, ' / ', training_epochs, 'cost:', avg_cost)

        #pbar.close()
        print("Optimization finished!")
        #minLoss = sess.run(tf.reduce_min(tmpResult))
        #min_reconstruction_loss = sess.run(tf.reduce_min(tmpResult2))
        #min_sparsity_loss = sess.run(tf.reduce_min(tmpResult3))
        #max_acc = sess.run(tf.reduce_max(tmpResult4))
        #min_sparsity_loss = sess.run(tf.reduce_min(tmpResult3))

        #print(minLoss)
        #minLoss = min(tmpResult) #this returns nan when values are zero
        #print(tmpResult.index(m))
        
    stop = timeit.default_timer()
    print('Time to run all training (sec): ', stop - start)
    print('Time to run each k-fold step (sec): ', (stop - start)/split_size)
    print('Time to run each epoch, average (sec): ', (stop - start)/training_epochs*total_batch)
    #reset tensorflow session
    tf.reset_default_graph()
    sess.close()
    # return the minimum loss of this combination of L1, L2, activation function, beta, rho on a dataset
    #return minLoss, min_reconstruction_loss, min_sparsity_loss, max_correl
    return mean_cost, my_rloss, mean_sloss, mean_acc, mean_cacc


# In[ ]:


def cross_validate(data_masked, data_obs, split_size=10):
    results = []
    kf = KFold(n_splits=split_size)
    for train_idx, val_idx in kf.split(data_masked, data_obs):
        train_x = data_masked[train_idx]
        train_y = data_obs[train_idx]
        val_x = data_masked[val_idx]
        val_y = data_obs[val_idx]
        session = run_autoencoder(lr, training_epochs, l1, l2, act, beta, rho, data_masked, keep_rate, data_obs)
        results.append(session.run(accuracy, feed_dict={x: val_x, y: val_y}))
    return results


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
        data_masked = mask_data(np.copy(data_obs),mask_rate,sys.argv[3])
        
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
            my_cost, my_rloss, my_sloss, my_acc, my_cacc = run_autoencoder(lr, training_epochs, l1, l2, act, beta, rho, data_masked, keep_rate, data_obs)
            tmpResult = [my_cost, my_rloss, my_sloss, my_acc, my_cacc]
            tmp_list.extend(tmpResult)
            result_list.append(tmp_list)
            i += 1
            print("TMP_RESULT: ", tmp_list)
        return result_list
            
    else:   
        data_obs = process_data(sys.argv[1],sys.argv[8]) #input file, i.e: HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf
        data_masked = mask_data(np.copy(data_obs),0.9,sys.argv[8])
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

        my_cost, my_rloss, my_sloss, my_acc, my_cacc = run_autoencoder(lr, training_epochs, l1, l2, act, beta, rho, data_masked, keep_rate, data_obs)

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

