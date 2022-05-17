# Train with Race
import numpy as np
import tensorflow as tf
import sys
import argparse
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import utils
import importlib
import os
from models import make_avazu_embedding_model, make_clickthrough_nn, make_criteo_embedding_model, make_movielens_embedding_model
from lsh_functions import PStableHash
from race import Race
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training params
    parser.add_argument("--epoch", action="store", default=4, type=int)
    parser.add_argument("--batch_size", action="store", default=512, type=int)
    parser.add_argument("--eval_step", action="store", default=5000, type=int)
    parser.add_argument("--lr", action="store", default=0.001, type=float)
    parser.add_argument("--h", action="store", default=800, type=int, help='hidden layer size')
    parser.add_argument("--n", action="store", default=4, type=int, help='number of hidden layers')
    parser.add_argument("--gpu", action="store", required=True, type=int)
    parser.add_argument("--seed", action="store", default=314150, type=int)
    parser.add_argument("--data", action="store", default = 'criteo', type=str)
    # RACE hyper parameters
    parser.add_argument("--r", action="store", default=100, type=int, help='repetitions')
    parser.add_argument("--c", action="store", default=1, type=int, help='concatenations')
    parser.add_argument("--b", action="store", default=10000, type=int, help='buckets')
    parser.add_argument("--p", action="store", default=2.0, type=float, help='value of p for the Lp norm space')
    # Weighting function hyper parameters
    parser.add_argument("--first_n", action="store", default=10000, type=int, help='accept fist n data points regardless')
    parser.add_argument("--thrsh", action="store", required=True, type=float, help='score threshold')
    parser.add_argument("--prob", action="store", required=True, type=float, help='accept probability')
    
    args = parser.parse_args()
    
    
    gpu_ind = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
    if gpu_ind != -1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

    seed = args.seed
    batch_size = args.batch_size
    lr = args.lr
    eval_step = args.eval_step
    n_epoch = args.epoch
    # RACE hyper parameters
    repetitions = args.r
    concatenations = args.c
    buckets = args.b
    p = args.p
    # Weighting function hyper parameters
    accept_first_n = args.first_n
    score_threshold = args.thrsh
    accept_prob = args.prob

    
    # timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    # rslt_dir = './final_results/rslt_end2end_train_race_thresh' + args.thrsh + '_accept' + args.prob + '_' + timestr
    # os.makedirs(rslt_dir)

    if args.data=='criteo':
        train_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/data/train.csv')
        val_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/data/valid.csv')
        test_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/data/test.csv')
        make_embedding_model = make_criteo_embedding_model

    if args.data=='avazu':
        train_ds = utils.load_avazu_csv('/Users/benitogeordie/Downloads/Avazu_x4/data/train_contig_noid.csv')
        val_ds = utils.load_avazu_csv('/Users/benitogeordie/Downloads/Avazu_x4/data/valid_contig_noid.csv')
        test_ds = utils.load_avazu_csv('/Users/benitogeordie/Downloads/Avazu_x4/data/test_contig_noid.csv')
        make_embedding_model = make_avazu_embedding_model

    if args.data=='movielens':
        train_ds = utils.load_movielens_csv('/Users/benitogeordie/Downloads/Movielenslatest_x1/data/train_contig.csv')
        val_ds = utils.load_movielens_csv('/Users/benitogeordie/Downloads/Movielenslatest_x1/data/valid_contig.csv')
        test_ds = utils.load_movielens_csv('/Users/benitogeordie/Downloads/Movielenslatest_x1/data/test_contig.csv')
        make_embedding_model = make_movielens_embedding_model
        


    train_ds_batch = train_ds.batch(batch_size)
    train_ds_batch = train_ds_batch.prefetch(2)
    batch_data_val = val_ds.batch(batch_size)
    batch_data_test = test_ds.batch(batch_size)

    race_embedding_model = make_embedding_model()
    hash_module = PStableHash(race_embedding_model.output_shape[1], num_hashes=repetitions * concatenations, p=p, seed=seed)
    race = Race(repetitions, concatenations, buckets, hash_module)

    weight_fn = utils.weight_with_race(race, race_embedding_model, accept_first_n, score_threshold, accept_prob)
    filtered_weighted_train_ds = utils.weight_and_filter(train_ds_batch, weight_fn)

    # Dimensions of neural network hidden layers.
    nn_embedding_model = make_embedding_model()
    hidden_layer_dims = [args.h]*args.n
    nn = make_clickthrough_nn(nn_embedding_model, hidden_layer_dims, lr)

    # _ = nn.fit(filtered_weighted_train_ds)
    # nn_embedding_model.compile('adam')
    # nn_embedding_model.evaluate(batch_data_val)
    # _ = nn.evaluate(batch_data_val)
    _ = nn.evaluate(filtered_weighted_train_ds)
