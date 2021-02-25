# coding: utf-8

import argparse
from datetime import date, datetime
import logging
import math
import multiprocessing as mp
from numba import cuda
import numpy as np
import os
import pandas as pd
import pathlib
import sys
import tensorflow as tf
import timeit
import warnings

from configure_logging import configure_logging, set_logger_to_debug
from genotype_conversions import convert_gt_to_int


logger = logging.getLogger("ImputationInference")


def set_gpu(use_gpu):
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def log_gpu_memory():
    for gpu in cuda.gpus:
        with gpu:
            memory_info = cuda.current_context().get_memory_info()
            logger.info(
                "GPU Device: %s, free memory: %s bytes, total, %s bytes" % (
                    gpu.name.decode("utf-8"), memory_info[0], memory_info[1])
            )


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
                        help="model name to load which is located under model_dir , \
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


@cuda.jit(device=True)
def get_predicted_gen_known_indices(new_df_sample, i, j, out_array):
    if new_df_sample[i, j] == 0:
        # gen = "1/1:2:0:0:1:0:1"
        out_array[i, j // 2, 0] = 1
        out_array[i, j // 2, 1] = 1
        out_array[i, j // 2, 2] = 2
        out_array[i, j // 2, 3] = 0
        out_array[i, j // 2, 4] = 0
        out_array[i, j // 2, 5] = 1
        out_array[i, j // 2, 6] = 0
        out_array[i, j // 2, 7] = 1
    elif new_df_sample[i, j + 1] == 0:
        # gen = "0/0:0:1:0:0:1:0"
        out_array[i, j // 2, 0] = 0
        out_array[i, j // 2, 1] = 0
        out_array[i, j // 2, 2] = 0
        out_array[i, j // 2, 3] = 1
        out_array[i, j // 2, 4] = 0
        out_array[i, j // 2, 5] = 0
        out_array[i, j // 2, 6] = 1
        out_array[i, j // 2, 7] = 0
    elif new_df_sample[i, j + 1] == 1 and new_df_sample[i, j] == 1:
        # gen = "0/1:1:0:1:0:1:1"
        out_array[i, j // 2, 0] = 0
        out_array[i, j // 2, 1] = 1
        out_array[i, j // 2, 2] = 1
        out_array[i, j // 2, 3] = 0
        out_array[i, j // 2, 4] = 1
        out_array[i, j // 2, 5] = 0
        out_array[i, j // 2, 6] = 1
        out_array[i, j // 2, 7] = 1


@cuda.jit("void(float64[:,:], float64[:,:], int64[:], float64[:,:,:])")
def compute_probabilites_infered(d_newdf_sample, y_pred_sample, known_indexes, probs_array):
    i, j = cuda.grid(2)
    if i < y_pred_sample.shape[0] and j < y_pred_sample.shape[1] and (j % 2) == 0:
        for idx in range(known_indexes.shape[0]):
            if j == known_indexes[idx]:
                get_predicted_gen_known_indices(d_newdf_sample, i, j, probs_array)
                return
        # Homo Ref = Ref * (1-ALT)
        # Het = Ref * Alt
        # Homo Alt = (1-Ref) * Alt
        Pa = y_pred_sample[i, j] * (1 - y_pred_sample[i, j + 1])
        Pab = y_pred_sample[i, j] * y_pred_sample[i, j + 1]
        Pb = (1 - y_pred_sample[i, j]) * y_pred_sample[i, j + 1]

        # P = np.array([Pa, Pab, Pb], dtype=np.float64)
        # P = [Pa, Pab, Pb]

        # 2 versions of dosage
        # Df0=(1-Pa)+Pb
        Df1 = Pab + (2 * Pb)
        if Df1 != 0:
            Df1 = Df1 / (Pa + Pab + Pb)

        # np.round(Df1,4,Df1)
        Df1 = Df1*(10**5)//1/(10**5)

        if Pa >= Pab and Pa >= Pb:
            D = 0
            probs_array[i, j // 2, 0] = probs_array[i, j // 2, 1] = 0
        elif Pab >= Pa and Pab >= Pb:
            D = 1
            probs_array[i, j // 2, 0] = 0
            probs_array[i, j // 2, 1] = 1
        elif Pb >= Pa and Pb >= Pab:
            D = 2
            probs_array[i, j // 2, 0] = probs_array[i, j // 2, 1] = 1

        probs_array[i, j//2, 2] = Df1
        probs_array[i, j//2, 3] = Pa
        probs_array[i, j//2, 4] = Pab
        probs_array[i, j//2, 5] = Pb
        probs_array[i, j//2, 6] = y_pred_sample[i, j]    # debug
        probs_array[i, j//2, 7] = y_pred_sample[i, j+1]  # debug

        # probs_array[j, 4] = np.float(round(Df1, 4))
        # Pa=np.round(Pa,4)
        # Pab=np.round(Pab,4)
        # Pb=np.round(Pb,4)


def create_predicted_gen(new_df_sample, known_indexes, y_pred_sample, is_debug_on, ret_out_sample):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(y_pred_sample.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(y_pred_sample.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    logger.info(
        "Copying %d bytes to GPU memory before invoking compute_probabilites_infered()" % (
                new_df_sample.nbytes + y_pred_sample.nbytes + known_indexes.nbytes + ret_out_sample.nbytes
        )
    )
    d_newdf_sample = cuda.to_device(new_df_sample)
    d_y_pred_sample = cuda.to_device(y_pred_sample)
    d_known_indexes = cuda.to_device(known_indexes)
    d_ret_out_sample = cuda.to_device(ret_out_sample)
    compute_probabilites_infered[blockspergrid, threadsperblock](
        d_newdf_sample, d_y_pred_sample, d_known_indexes, d_ret_out_sample
    )
    return d_ret_out_sample.copy_to_host()


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


def run_inference(meta_path, model_path, new_df):
    config = tf.ConfigProto(log_device_placement=False)
    # config.intra_op_parallelism_threads = 24
    # config.inter_op_parallelism_threads = 1
    # config.gpu_options.per_process_gpu_memory_fraction = 0.15
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(str(meta_path))
        saver.restore(sess, str(model_path))
        y_pred = (sess.run('y_pred:0', feed_dict={"X:0": new_df, "Y:0": new_df}))
        # y_pred_prob = np.copy(y_pred)
        # convert presence probabilities into allele counts : [0,2],[2,0],[1,1]
        # y_pred = sess.run(tf.round( tf.cast(y_pred, tf.float64)))
        y_pred = sess.run(tf.cast(y_pred, tf.float64))
    return y_pred


def vcf_formatted_output(arr, is_debug):
    subjects_formatted = []
    for j in range(len(arr)):
        subjects_formatted.append(
            str(arr[j, 0].astype(int)) + "/" + str(arr[j, 1].astype(int)) + ":" +
            str(arr[j, 2]) + ":" + str(arr[j, 3]) + ":" + str(arr[j, 4]) + ":" + str(arr[j, 5]) +
            (":" + str(arr[j, 6]) + ":" + str(arr[j, 7]) if is_debug else "")
        )
    return subjects_formatted


def main(argv):
    args = parse_arguments(argv)
    logger_file_name = "inference_{}.log".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    configure_logging(logger, args.output.with_name(logger_file_name))
    logger.info("Parsed arguments: %s" % args)
    if args.debug:
        set_logger_to_debug(logger)
    set_gpu(args.use_gpu)
    meta_path = args.model_dir.joinpath(args.model_name + ".ckpt.meta")
    model_path = args.model_dir.joinpath(args.model_name + ".ckpt")
    if not meta_path.is_file() or not model_path.with_suffix(".ckpt.index").is_file():
        logger.error("No model files (.ckpt.meta/.ckpt) were found under %s which begin with the name: %s" %
                     (args.model_dir, args.model_name))
        raise RuntimeError("Could not load TF model")

    start = timeit.default_timer()
    new_df, known_indexes = process_data(args.posfile, args.infile)
    new_df = flatten(new_df.copy())
    log_gpu_memory()
    logger.info('Time to load & preprocess the data (sec): %f' % (timeit.default_timer() - start))

    start = timeit.default_timer()
    # Executing inference in a different process to clear GPU memory after Session ends
    with mp.Pool(processes=1) as pool:
        y_pred = pool.apply(run_inference, (meta_path, model_path, new_df))
    log_gpu_memory()
    logger.info('Time to reload model & do inference (sec): %f' % (timeit.default_timer() - start))

    start = timeit.default_timer()
    out = np.empty([y_pred.shape[0], y_pred.shape[1] // 2, 8], dtype=np.float64)
    output_array = create_predicted_gen(new_df, known_indexes, y_pred, args.debug, out)
    logger.info('Time to create predicted gen output data (sec): %f' % (timeit.default_timer() - start))

    start = timeit.default_timer()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        str_output_array = pool.starmap(
            vcf_formatted_output,
            [(output_array[i], args.debug) for i in range(output_array.shape[0])]
        )
    logger.info('Time to convert output to vcf format(sec): %f' % (timeit.default_timer() - start))

    start = timeit.default_timer()
    export_vcf(args.posfile, str_output_array, args.infile, args.output)
    logger.info('Time to write output file (sec): %f' % (timeit.default_timer() - start))


if __name__ == "__main__":
    # Required to run CUDA with multiprocessing
    # https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    mp.set_start_method('spawn', force=True)
    main(sys.argv[1:])
