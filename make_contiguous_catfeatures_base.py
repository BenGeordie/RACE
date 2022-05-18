# Make criteo dataset with columns with contiguous numbers 
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pdb

# TODO input file name
# TODO output file prefix
# TODO start at

def make_contiguous(small_csv, train_csv, test_csv, valid_csv, start_idx, out_prefix, use_gpu, hex_col_idxs=[]):
    if use_gpu:
        gpu_ind = 1
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

    dta = pd.read_csv(small_csv) # load a very small version of data to only save its columns
    cols = dta.columns
    level=[]
    new_df_tr = pd.DataFrame(columns=cols)
    new_df_val = pd.DataFrame(columns=cols)
    new_df_test = pd.DataFrame(columns=cols)

    vocab_sizes = []

    for i,col in enumerate(cols):
        dta_col_tr = pd.read_csv(train_csv,usecols = [col])
        dta_col_test = pd.read_csv(test_csv,usecols = [col])
        dta_col_val = pd.read_csv(valid_csv,usecols = [col])
        
        if i in hex_col_idxs:
            dta_col_tr[col] = dta_col_tr[col].apply(int, base=16)
            dta_col_test[col] = dta_col_test[col].apply(int, base=16)
            dta_col_val[col] = dta_col_val[col].apply(int, base=16)
        
        print("COLUMN", i)
        print("train preview")
        print(dta_col_tr.head(5))
        print("test preview")
        print(dta_col_test.head(5))
        print("val preview")
        print(dta_col_val.head(5))

        if i < start_idx:
            new_df_tr[col]=dta_col_tr[col]
            new_df_val[col]=dta_col_val[col]
            new_df_test[col]=dta_col_test[col]
        else:
            dta_col_tr[col]=dta_col_tr[col].apply(lambda x: x + 1) # remove -1's if any
            dta_col_val[col]=dta_col_val[col].apply(lambda x: x + 1) # remove -1's if any
            dta_col_test[col]=dta_col_test[col].apply(lambda x: x + 1) # remove -1's if any

            vocab = dta_col_tr[col].unique()

            vocab_sizes.append(len(vocab))

            data_tr = tf.constant([dta_col_tr[col]])
            data_val = tf.constant([dta_col_val[col]])
            data_test = tf.constant([dta_col_test[col]])
            layer = tf.keras.layers.IntegerLookup(vocabulary=vocab,num_oov_indices=1)
            
            new_df_tr[col]=layer(data_tr).numpy().flatten()
            new_df_test[col]=layer(data_test).numpy().flatten()
            new_df_val[col]=layer(data_val).numpy().flatten()

    new_df_tr.to_csv(out_prefix + 'train_contig.csv',index=False)
    new_df_test.to_csv(out_prefix + 'test_contig.csv',index=False)
    new_df_val.to_csv(out_prefix + 'valid_contig.csv',index=False)
    with open(out_prefix + "vocab_sizes.txt", 'w') as f:
        f.write('[')
        f.write(', '.join([str(size) for size in vocab_sizes]))
        f.write(']')

