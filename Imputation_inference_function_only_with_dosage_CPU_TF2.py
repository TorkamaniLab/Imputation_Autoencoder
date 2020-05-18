# coding: utf-8

import argparse
from datetime import date, datetime
import inspect
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pathlib
import sys
import tensorflow as tf
import timeit
import warnings

from configure_logging import configure_logging, set_logger_to_debug, elapsed_filter
from genotype_conversions import convert_gt_to_int

from DSAE_TF2_keras_7_checkpoint import kl_divergence_regularizer, Rsquared
import tensorflow_addons as tfa

logger = logging.getLogger("ImputationInference")


def set_gpu(use_gpu):
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def is_valid_path(parser, arg, valid_extensions=[], is_output_file=False):
    arg_path = pathlib.Path(arg)
    if valid_extensions and not any([arg_path.match('*.' + ext) for ext in valid_extensions]):
        warnings.warn('File %s does not end with any of the known extensions %s' % (arg, valid_extensions))
    if not is_output_file:
        to_resolve_path = arg_path
    else:
        to_resolve_path = arg_path.parents[0]
    try:
        _ = to_resolve_path.resolve(strict=True)
    except FileNotFoundError:
        parser.error('Arguments validation error: %s does not exist!' % to_resolve_path)
    return arg_path


def parse_arguments(user_arguments):
    parser = argparse.ArgumentParser(description='Imputation Inference Script.')
    parser.add_argument('posfile',
                        metavar='reference.1-5',
                        type=lambda x: is_valid_path(parser, x, valid_extensions=['1-5', 'pos']),
                        help="First 5 columns of reference panel VCF file (chromosome position rsID REF ALT),\
                        used to build imputed file")
    parser.add_argument('infile',
                        metavar='genotype_array',
                        type=lambda x: is_valid_path(parser, x, valid_extensions=['masked']),
                        help="Genotype array file in VCF format, file to be imputed")
    parser.add_argument('model_dir',
                        metavar='model_dir',
                        type=lambda x: is_valid_path(parser, x, valid_extensions=[]),
                        help="Pre-trained model directory path \
                        (just directory path, no file name and no extension required")
    parser.add_argument('--model_name',
                        metavar='model_name',
                        type=str,
                        required=True,
                        help="model name to load which is located under model_dir ,\
                                for multi-model support ")
    parser.add_argument('--output',
                        metavar='output',
                        type=lambda x: is_valid_path(parser, x, valid_extensions=[], is_output_file=True),
                        default=os.path.dirname(os.path.realpath(__file__)),
                        help="(Optional) An output file location (can be either a file or a directory),\
                        imputed file in VCF format, same name prefix as input if no out name is provided")
    parser.add_argument('--use_gpu',
                        action='store_true',
                        default=False,
                        help="Whether or not to use the GPU (default=False)")
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help="Add predicted accuracies to the output for debugging (default=False)")
    args = parser.parse_args(user_arguments)

    if pathlib.Path(args.output).is_dir():
        args.output = args.output.joinpath(args.infile.name + ".autoencoder_imputed.vcf")
    else:
        args.output = is_valid_path(parser, args.output, valid_extensions=[], is_output_file=True)

    return args


def get_positions_from_reference_file(posfile):
    # Header and column names start with hastag, skip those
    # posfile should contain 2 columns separated by tab: 1st = chromosome ID, 2nd = position
    # vcf can be imported as posfile as well, but will take much longer to read and process
    refpos = pd.read_csv(posfile, sep='\t', comment='#', header=None)
    # 0      22065657
    # 1      22065697
    # 2      22065904
    # 3      22065908
    # 4      22065974
    # 5      22065977
    if refpos.shape[1] != 5:
        logger.error("postfile is not structured as 5 columns (%d)" % refpos.shape[1])
        raise RuntimeError("Unexpected postfile structure")
    posfile_nucleotides = set(refpos[3].unique()).union(refpos[4].unique())
    unexpected_nucleotides = posfile_nucleotides - {'A', 'C', 'G', 'T'}
    if unexpected_nucleotides:
        logger.error("Unexpected nucleotides values were found in 4th & 5th columns %s" % unexpected_nucleotides)
        raise RuntimeError("Unexpected postfile nucleotides values")
    return pd.Series(refpos[1], index=range(len(refpos[1])))


def process_data(posfile, infile):

    refpos = get_positions_from_reference_file(posfile)

    #infile is the input file: genotype data set to be imputed
    df = pd.read_csv(infile, sep='\t', comment='#', header=None)

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

    new_df = np.zeros((df.shape[1]-9, len(refpos), 2))  # subjects, variants

    logger.info("Processing input data")

    _, _, known_indexes = np.intersect1d(inpos.index.values, refpos.values, return_indices=True)
    known_indexes = 2 * known_indexes
    logger.info("Adding known variant {} to known list to exclude from accuracy calculations.".format(known_indexes))

    i = 9  # RR column index
    idx = 0
    inpos_keys = inpos.keys()
    while i < df.shape[1]:
        for refpos_idx, refpos_val in refpos.iteritems():
            if refpos_val in inpos_keys:
                myidx = inpos[refpos_val]
                new_df[idx][refpos_idx] = convert_gt_to_int(df[i][myidx][0:3])
                # To prevent string evaluation overhead
                if logger.level == 10:  # If debug level is set to 'DEBUG'
                    logger.debug("%s 0-> %s" % (df[i][myidx][0:3], new_df[idx][refpos_idx]))
        i += 1
        idx += 1
    #the data needs to be flattened because the matrix multiplication step (x*W)
    #doesn't support features with subfeatures (matrix of vectors)
    #new_df = np.reshape(new_df, (new_df.shape[0],new_df.shape[1]*2))
    #print(new_df.shape)
    #pbar.close()
    return new_df, known_indexes


def flatten(mydata):
    # subjects, SNP, REF/ALT counts
    if len(mydata.shape) == 3:
        mydata = np.reshape(mydata, (mydata.shape[0], -1))
    else:  # do one hot encoding, depth=3 because missing (-1) is encoded to all zeroes
        mydata = tf.one_hot(indices=mydata, depth=3)
        mydata = tf.layers.flatten(mydata)  # flattening to simplify calculations later (matmul, add, etc)
    return mydata


# split input data into chunks so we can prepare batches in parallel
def chunk(L, nchunks):
    L2 = list()
    j = round(len(L)/nchunks)
    chunk_size = j
    i = 0
    while i < len(L):
        chunk = L[i:j]
        L2.append(chunk)
        i = j
        j += chunk_size
        if j > len(L):
            j = len(L)
    return L2


def get_predicted_gen_known_indices(new_df_sample, j):
    if new_df_sample[j] == 0:
        return "1/1:2:0:0:1:0:1"
    elif new_df_sample[j + 1] == 0:
        return "0/0:0:1:0:0:1:0"
    elif new_df_sample[j + 1] == 1 and new_df_sample[j] == 1:
        return "0/1:1:0:1:0:1:1"
    plogger = logging.getLogger("ImputationInference").getChild(inspect.stack()[1][3])  # Extract caller function name
    plogger.error("Error in get_predicted_gen_known_indices() %s %s" % (new_df_sample[j], new_df_sample[j + 1]))
    return None


def get_predicted_gen_imputed(y_pred_sample, j, is_debug_on):
    # Homo Ref = Ref * (1-ALT)
    # Het = Ref * Alt
    # Homo Alt = (1-Ref) * Alt
    Pa = y_pred_sample[j] * (1 - y_pred_sample[j + 1])
    Pab = y_pred_sample[j] * y_pred_sample[j + 1]
    Pb = (1 - y_pred_sample[j]) * y_pred_sample[j + 1]
    P = [Pa, Pab, Pb]

    # 2 versions of dosage
    # Df0=(1-Pa)+Pb
    Df1 = Pab + (2 * Pb)
    if Df1 != 0:
        Psum = Pa + Pab + Pb
        Df1 = Df1 / Psum
    D = np.argmax(P)
    Df1 = np.round(Df1, 4)
    # Pa=np.round(Pa,4)
    # Pab=np.round(Pab,4)
    # Pb=np.round(Pb,4)
    if D == 2:
        gen = "1/1:"
    elif D == 0:
        gen = "0/0:"
    elif D == 1:
        gen = "0/1:"
    else:
        plogger = logging.getLogger("ImputationInference").getChild(inspect.stack()[1][3])  # Extract caller function name
        plogger.error("ERROR dosage = %d  %f%" % (D, P))
        return None
    gen += str(Df1) + ":" + str(Pa) + ":" + str(Pab) + ":" + str(Pb)
    if is_debug_on is True:
        gen += ":" + str(y_pred_sample[j]) + ":" + str(y_pred_sample[j + 1])
    return gen


def create_predicted_gen(new_df_sample, known_indexes, y_pred_sample, is_debug_on):
    # Extract & set the current function name for child logger
    j = 0
    ret_out = []
    while j < len(y_pred_sample):
        gen = None
        if j in known_indexes:
            gen = get_predicted_gen_known_indices(new_df_sample, j)
        # elif(y_pred_sample[j]!=0 or y_pred_sample[j+1]!=0):
        else:
            gen = get_predicted_gen_imputed(y_pred_sample, j, is_debug_on)
        # else:
        #    print("Warning, prediction with missing value: ", i, j,y_pred_sample[j], y_pred_sample[j+1])
        #    gen = "0/0:0"
        j += 2
        if gen is None:
            plogger = logging.getLogger("ImputationInference").getChild(inspect.stack()[0][3])
            plogger.addFilter(elapsed_filter)  # is not inherited from parent logger
            plogger.error("ERROR, genotype NULL. y_pred[i][j] %f" % y_pred_sample[j])
            raise RuntimeError("Genotype evaluation error")
        ret_out.append(gen)
    return ret_out


def get_header_line(input_file):
    with open(input_file) as f:
        for line in f:
            if line.startswith("#CHROM"):
                return line
    return None


def export_vcf(posfile, pred_out, infile, output_file):
    my_header = get_header_line(infile)
    if not my_header:
        raise Exception("Could not extract header line from genotype_array file")

    refpos = pd.read_csv(posfile, sep='\t', comment='#', header=None)

    pred_out = np.transpose(pred_out)
    refpos = np.asarray(refpos.values)

    comments = \
        "##fileformat=VCFv4.1\n##filedate=" + str(date.today()) + \
        "\n##source=Imputation_autoencoder\n##contig=<ID=" + str(refpos[0][0])+">\n" + \
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n" + \
        "##FORMAT=<ID=DS,Number=1,Type=Float,Description=\"Estimated Alternate Allele Dosage : [P(0/1)+2*P(1/1)]\">\n" + \
        "##FORMAT=<ID=Paa,Number=1,Type=Float,Description=\"Imputation probability for homozigous reference : Pa=y_pred[i][j]*(1-y_pred[i][j+1])\">\n" + \
        "##FORMAT=<ID=Pab,Number=1,Type=Float,Description=\"Imputation probability for heterozigous : Pab=y_pred[i][j]*y_pred[i][j+1]\">\n" + \
        "##FORMAT=<ID=Pbb,Number=1,Type=Float,Description=\"Imputation probability for homozigous alternative : Pb=(1-y_pred[i][j])*y_pred[i][j+1]\">\n" + \
        "##FORMAT=<ID=AP,Number=1,Type=Float,Description=\"Predicted presence of reference allele (autoencoder raw output)\">\n" + \
        "##FORMAT=<ID=BP,Number=1,Type=Float,Description=\"Predicted presence of alternative allele (autoencoder raw output)\">\n" + \
        my_header

    #print(refpos.shape)
    #print(len(refpos))
    #print(len(pred_out))

    columns = ['.', '.', '.', 'GT:DS:Paa:Pab:Pbb:AP:BP']*len(refpos)
    columns = np.reshape(columns, (len(refpos), 4))

    refpos = np.concatenate((refpos, columns), axis=1)
    vcf = np.concatenate((refpos, pred_out), axis=1)

    with open(output_file, 'w') as f:
        f.write(comments)  # python will convert \n to os.linesep
    with open(output_file, 'ab') as bf:  # open as binary
        np.savetxt(bf, vcf, delimiter="\t", fmt='%s')
    logger.info("RESULT: %s" % output_file)


def main(argv):
    args = parse_arguments(argv)
    logger_file_name = "inference_{}.log".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    configure_logging(logger, args.output.with_name(logger_file_name))
    logger.info("Parsed arguments: %s" % args)
    if args.debug:
        set_logger_to_debug(logger)

    set_gpu(args.use_gpu)

    meta_path = args.model_dir.joinpath(args.model_name + ".ckpt.meta")
    '''
    model_path = args.model_dir.joinpath(args.model_name + ".ckpt")
    if not meta_path.is_file() or not model_path.with_suffix(".ckpt.index").is_file():
        logger.error("No model files (.ckpt.meta/.ckpt) were found under %s which begin with the name: %s" %
                     (args.model_dir, args.model_name))
        raise RuntimeError("Could not load TF model")
    '''
    model_path = args.model_dir.joinpath(args.model_name)

    start = timeit.default_timer()
    new_df, known_indexes = process_data(args.posfile, args.infile)
    new_df = flatten(new_df.copy())
    logger.info('Time to load & preprocess the data (sec): %f' % (timeit.default_timer() - start))

    '''
    config = tf.ConfigProto(log_device_placement=False)
    config.intra_op_parallelism_threads = 24
    config.inter_op_parallelism_threads = 1
    #config.gpu_options.per_process_gpu_memory_fraction = 0.15
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    '''

    tf.compat.v1.ConfigProto(log_device_placement=False)
    tf.config.threading.set_intra_op_parallelism_threads(24)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    # Allow memory growth for the GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for GPU in physical_devices:
        tf.config.experimental.set_memory_growth(GPU, True)

    '''
    with tf.Session(config=config) as sess:
        start = timeit.default_timer()
        saver = tf.train.import_meta_graph(str(meta_path))
        saver.restore(sess, str(model_path))
        y_pred = (sess.run('y_pred:0', feed_dict={"X:0": new_df, "Y:0": new_df}))
        # y_pred_prob = np.copy(y_pred)
        # convert presence probabilities into allele counts : [0,2],[2,0],[1,1]
        # y_pred = sess.run(tf.round( tf.cast(y_pred, tf.float64)))
        y_pred = sess.run(tf.cast(y_pred, tf.float64))
        logger.info('Time to reload model & do inference (sec): %f' % (timeit.default_timer() - start))
    '''
    loaded_model = tf.keras.models.load_model(model_path,  custom_objects={ 'kl_divergence_regularizer': kl_divergence_regularizer(), 'Rsquared': Rsquared, 'F1Score': tfa.metrics.F1Score}, compile=False)
    #loaded_model = tf.keras.models.load_model(model_path,  custom_objects={ 'kl_divergence_regularizer': kl_divergence_regularizer()}, compile=False)
    #loaded_model.compile(optimizer='Adam', loss='MSE', metrics='accuracy')#toy options, no training will be done, so they should not make a difference
    y_pred = loaded_model.predict(new_df)
    logger.info('Time to reload model & do inference (sec): %f' % (timeit.default_timer() - start))

    start = timeit.default_timer()
    pool = mp.Pool(mp.cpu_count())
    out = pool.starmap(
        create_predicted_gen,
        [(new_df[i], known_indexes, y_pred[i], args.debug) for i in range(len(y_pred))]
    )
    pool.close()
    logger.info('Time to prepare output data (sec): %f' % (timeit.default_timer() - start))

    start = timeit.default_timer()
    export_vcf(args.posfile, out, args.infile, args.output)
    logger.info('Time to write output file (sec): %f' % (timeit.default_timer() - start))


if __name__ == "__main__":
    main(sys.argv[1:])
