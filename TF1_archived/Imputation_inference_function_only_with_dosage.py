
# coding: utf-8

# In[10]:


import numpy as np
#from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
from datetime import date

import itertools

import pandas as pd

# sorting results
from collections import defaultdict
from operator import itemgetter
import timeit #measure runtime
import random #masking
import multiprocessing
import operator
from functools import partial
#import allel
#from scipy.spatial.distance import squareform
import sys
import os

# In[11]:
#model_dir = "/home/rdias/myscripts/raqueld/Autoencoder_tensorflow/test_new_grid_search_with_FL_5cycles/"

#posfile = "HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf.clean4.pos.1-5"
#infile = "ARIC_PLINK_flagged_chromosomal_abnormalities_zeroed_out_bed.lifted_NCBI36_to_GRCh37.GH.ancestry-1.chr9_intersect1.vcf.gz.9p21.3.recode.vcf"
DEBUG=False
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

if(len(sys.argv)>=4):

    posfile = sys.argv[1]
    infile =  sys.argv[2]
    model_dir = sys.argv[3]

else:

   print("\nUsage: python3 Imputation_inference_function.py reference.1-5 genotype_array model_file output_file_name model_index\n")
   print("       reference.1-5:       first 5 columns of reference panel VCF file (chromosome position rsID REF ALT), used to build imputed file")
   print("       genotype_array:      genotype array file in VCF format, file to be imputed")
   print("       model_file:          pretrained model directory path (just directory path, no file name and no extension required")
   print("       output_file_name:    (optional) a name for the output file, imputed file in VCF format, same name prefix as input if no out name is provided")
   print("       model_index:    (optional) if validating grid search results, provide model index, integer starting from 0")

   sys.exit()

import tensorflow as tf


meta_path = model_dir+"/inference_model-best.ckpt.meta"
model_path = model_dir+"/inference_model-best.ckpt"



#included multi-model support
if(len(sys.argv)==6):
    model_index=sys.argv[5]
    meta_path = model_dir+"/"+model_index+".meta"
    model_path = model_dir+"/"+model_index



def convert_gt_to_int(gt,alt_signal_only=False):

    genotype_to_int={'0/0': [1,0], '0|0': [1,0], '0/1': [1,1], '0|1':[1,1], '1/0':[1,1], '1|0':[1,1], '1/1':[0,1], '1|1':[0,1], './0':[0,0], './1':[0,0], './.':[0,0], '0/.':[0,0], '1/.':[0,0]}
    result=genotype_to_int[gt[0:3]]

    if(alt_signal_only==True):
        genotype_to_int={'0/0': 0, '0|0': 0.0, '0/1': 1, '0|1':1, '1/0':1, '1|0':1, '1/1':2, '1|1':2, './0':-1, './1':-1, './.':-1, '0/.':-1, '1/.':-1}
    
    return result

def process_data(posfile, infile):

    start = timeit.default_timer()

    #Header and column names start with hastag, skip those
    #posfile should contain 2 columns separated by tab: 1st = chromosome ID, 2nd = position
    #vcf can be imported as posfile as well, but will take much longer to read and process
    refpos = pd.read_csv(posfile, sep='\t', comment='#',header=None)

    #0      22065657
    #1      22065697
    #2      22065904
    #3      22065908
    #4      22065974
    #5      22065977

    refpos = pd.Series(refpos[1], index=range(len(refpos[1])))

    #print(refpos[1])

    #infile is the input file: genotype data set to be imputed
    df = pd.read_csv(infile, sep='\t', comment='#',header=None)

    #0      22065657
    #1      22066211
    #2      22066363
    #3      22066572
    #4      22067004
    #5      22067276

    inpos = pd.Series(range(len(df[1])), index=df[1])

    #print(inpos[2])
    #genetic variants are rows and samples are columns
    #let's transpose so the variants become columns and samples are rows
    df_T = df.transpose()

    new_df = np.zeros((len(df_T)-9,len(refpos),2))  #subjects, variants

    #print(new_df.shape)
    i = 9 #RR column index
    j = 0 #RR row index
    idx = 0
    print("Processing input data.")
    myidx = 0


    known_indexes = []
    k=0
    for j in range(len(refpos)):
        if(refpos[j] in inpos.keys()):
            print("Adding known variant", k, "to known list to exclude from accuracy calculations.")
            known_indexes.append(k)
        k=k+2

    while i < len(df_T):
        j = 0
        while j < len(refpos): #"|" is present when phased data is proved, "/" is usually unphased
            if(refpos[j] in inpos.keys()):
                myidx = inpos[refpos[j]]
                new_df[idx][j]=convert_gt_to_int(df[i][myidx][0:3])
                if(DEBUG==True):
                    print(df[i][myidx][0:3], "O->", new_df[idx][j])
            else:
                new_df[idx][j]=convert_gt_to_int('./.')
                if(DEBUG==True):
                    print(df[i][myidx][0:3], "X->", new_df[idx][j])
            j += 1

        i += 1
        idx += 1

    #the data needs to be flattened because the matrix multiplication step (x*W) 
    #doesn't support features with subfeatures (matrix of vectors)
    #new_df = np.reshape(new_df, (new_df.shape[0],new_df.shape[1]*2))
    #print(new_df.shape)
    #pbar.close()
    stop = timeit.default_timer()

    print('Time to load the data (sec): ', stop - start)
    
    return new_df, known_indexes


# In[12]:


def flatten(mydata):
    #subjects, SNP, REF/ALT counts
    if(len(mydata.shape) == 3):
        mydata = np.reshape(mydata, (mydata.shape[0],-1))
    else:#do one hot encoding, depth=3 because missing (-1) is encoded to all zeroes
        mydata = tf.one_hot(indices=mydata, depth=3)
        mydata = tf.layers.flatten(mydata)#flattening to simplify calculations later (matmul, add, etc)
    return mydata


# In[15]:


#split inut data into chunks so we can prepare batches in parallel
def chunk(L,nchunks):
    L2 = list()
    j = round(len(L)/nchunks)
    chunk_size = j
    i = 0
    while i < len(L):
        chunk = L[i:j]
        L2.append(chunk)
        i = j
        j += chunk_size
        if(j>len(L)):
            j = len(L)
    return L2


def export_vcf(posfile, pred_out, infile):



    with open(infile) as f:
        for line in f:
            if(line.startswith("#CHROM")):
                my_header=line
                break

    refpos = pd.read_csv(posfile, sep='\t', comment='#',header=None)

    pred_out = np.transpose(pred_out)
    refpos = np.asarray(refpos.values)

    comments="##fileformat=VCFv4.1\n##filedate="+str(date.today())+"\n##source=Imputation_autoencoder\n##contig=<ID="+str(refpos[0][0])+">\n"
    comments=comments + "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
    comments=comments + "##FORMAT=<ID=DS,Number=1,Type=Float,Description=\"Estimated Alternate Allele Dosage : [P(0/1)+2*P(1/1)]\">\n"
    comments=comments + "##FORMAT=<ID=Paa,Number=1,Type=Float,Description=\"Imputation probability for homozigous reference : Pa=y_pred[i][j]*(1-y_pred[i][j+1])\">\n"
    comments=comments + "##FORMAT=<ID=Pab,Number=1,Type=Float,Description=\"Imputation probability for heterozigous : Pab=y_pred[i][j]*y_pred[i][j+1]\">\n"
    comments=comments + "##FORMAT=<ID=Pbb,Number=1,Type=Float,Description=\"Imputation probability for homozigous alternative : Pb=(1-y_pred[i][j])*y_pred[i][j+1]\">\n"
    comments=comments + "##FORMAT=<ID=AP,Number=1,Type=Float,Description=\"Predicted presence of reference allele (autoencoder raw output)\">\n"
    comments=comments + "##FORMAT=<ID=BP,Number=1,Type=Float,Description=\"Predicted presence of alternative allele (autoencoder raw output)\">\n"
    comments=comments + my_header

    #print(refpos.shape)
    #print(len(refpos))
    #print(len(pred_out))



    columns = ['.','.','.','GT:DS:Paa:Pab:Pbb:AP:BP']*len(refpos)
    columns = np.reshape( columns, (len(refpos),4))
    #print(columns.shape)

    refpos = np.concatenate((refpos,columns), axis=1)
    vcf = np.concatenate((refpos,pred_out), axis=1)

    if(len(sys.argv)<5):
         outfile = infile + ".autoencoder_imputed.vcf"
    else:
         outfile = sys.argv[4]


    f = open(outfile, 'w')
    f.write(comments)  # python will convert \n to os.linesep
    f.close()

    out=open(outfile,'ab') #open as binary

    np.savetxt(out, vcf, delimiter="\t", fmt='%s')

    out.close()

    print("RESULT:",outfile)

new_df, known_indexes = process_data(posfile, infile)


new_df = flatten(new_df.copy())


config = tf.ConfigProto(log_device_placement=False)
config.intra_op_parallelism_threads = 24
config.inter_op_parallelism_threads = 1
#config.gpu_options.per_process_gpu_memory_fraction = 0.15
config.gpu_options.allow_growth=True
    
tf.reset_default_graph()


with tf.Session(config=config) as sess:

    start = timeit.default_timer()

    saver = tf.train.import_meta_graph(meta_path)

    saver.restore(sess,model_path)
    #for g_var in tf.global_variables():
    #    print(g_var)
    #for g_var in tf.local_variables():
    #    print(g_var)

    #graph = sess.graph

    #a=[n.name for n in graph.as_graph_def().node]
    #print(a)
    #optimizer = graph.get_operation_by_name( "optimizer" )
    #print(sess.run('Y:0', feed_dict={"X:0": new_df, "Y:0": new_df}))
    #print(sess.run('X:0', feed_dict={"X:0": new_df, "Y:0": new_df}))
    #print("\n****\n")
    #print(new_df)
    y_pred = (sess.run('y_pred:0', feed_dict={"X:0": new_df, "Y:0": new_df}))
    #print("\n****\n")
    #print(y_pred)
    stop = timeit.default_timer()
    print('Time to do inference (sec): ', stop - start)
    
    start = timeit.default_timer()
    #y_pred_prob = np.copy(y_pred)
    #convert presence probabilities into allele counts : [0,2],[2,0],[1,1]
#    y_pred = sess.run(tf.round( tf.cast(y_pred, tf.float64)))
    y_pred = sess.run(tf.cast(y_pred, tf.float64))

    out = []
    for i in range(len(y_pred)):
        j=0
        k=0
        tmp_out = []
        while j < len(y_pred[i]):
            gen = "NULL"
            if(j in known_indexes):
                if(new_df[i][j] == 0):
                    gen = "1/1:2:0:0:1:0:1"
                elif(new_df[i][j+1] == 0):
                    gen = "0/0:0:1:0:0:1:0"
                elif(new_df[i][j+1] == 1 and new_df[i][j] == 1):
                    gen = "0/1:1:0:1:0:1:1"
                else:
                    print("EEEERRRRROOOOOORRRR", new_df[i][j], new_df[i][j+1])
            #elif(y_pred[i][j]!=0 or y_pred[i][j+1]!=0):
            else:

                #Homo Ref = Ref * (1-ALT)
                #Het = Ref * Alt
                #Homo Alt = (1-Ref) * Alt

                Pa=y_pred[i][j]*(1-y_pred[i][j+1])
                Pab=y_pred[i][j]*y_pred[i][j+1]
                Pb=(1-y_pred[i][j])*y_pred[i][j+1]
                P=[Pa,Pab,Pb]

                #2 versions of dosage
                #Df0=(1-Pa)+Pb
                Df1=Pab+(2*Pb)
                if(Df1!=0):
                    Psum=Pa+Pab+Pb
                    Df1=Df1/Psum

                D=np.argmax(P)
                Df1=np.round(Df1,4)
                #Pa=np.round(Pa,4)
                #Pab=np.round(Pab,4)
                #Pb=np.round(Pb,4)
                if(D==2):
                    gen = "1/1:"
                elif(D==0):
                    gen = "0/0:"
                elif(D==1):
                    gen = "0/1:"
                else:
                    print("ERROR dosage = ", D, P)
                gen=gen+str(Df1)+":"+str(Pa)+":"+str(Pab)+":"+str(Pb)
                if(DEBUG==True):
                    gen=gen+":"+str(y_pred[i][j])+":"+str(y_pred[i][j+1])
            #else:
            #    print("Warning, prediction with missing value: ", i, j, y_pred[i][j], y_pred[i][j+1]) 
            #    gen = "0/0:0"
            j = j+2
            k = k+1
            if(gen=="NULL"):
                print("ERROR, genotype NULL. y_pred[i][j]", y_pred[i][j])
                sys.exit()
            tmp_out.append(gen)
        out.append(tmp_out)

    np.asarray(out)

#    print("\n*****AFTER CONVERSION\n")
#    for i in range(0,len(out)):
#        print(out[i][0])
#    print("\n****\n")

    stop = timeit.default_timer()

    export_vcf(posfile, out, infile)

print('Time to write output file (sec): ', stop - start)


