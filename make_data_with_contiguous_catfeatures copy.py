# Make criteo dataset with columns with contiguous numbers 
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pdb



gpu_ind = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# TODO input file name
# TODO output file prefix
# TODO start at

dta = pd.read_csv('criteo_x1_small.csv') # load a very small version of data to only save its columns
cols = dta.columns
level=[]
new_df_tr = pd.DataFrame(columns=cols)
new_df_val = pd.DataFrame(columns=cols)
new_df_test = pd.DataFrame(columns=cols)
for i,col in enumerate(cols):
    dta_col_tr = pd.read_csv('train_shuffle.csv',usecols = [col])
    dta_col_test = pd.read_csv('test_shuffle.csv',usecols = [col])
    dta_col_val = pd.read_csv('valid_shuffle.csv',usecols = [col])
    if i<14:
        new_df_tr[col]=dta_col_tr[col]
        new_df_val[col]=dta_col_val[col]
        new_df_test[col]=dta_col_test[col]
    else:
        vocab = dta_col_tr[col].unique()
        data_tr = tf.constant([dta_col_tr[col]])
        data_val = tf.constant([dta_col_val[col]])
        data_test = tf.constant([dta_col_test[col]])
        layer = tf.keras.layers.IntegerLookup(vocabulary=vocab,num_oov_indices=1)
        
        new_df_tr[col]=layer(data_tr).numpy().flatten()
        new_df_test[col]=layer(data_test).numpy().flatten()
        new_df_val[col]=layer(data_val).numpy().flatten()

new_df_tr.to_csv('train_contig.csv',index=False)
new_df_test.to_csv('test_contig.csv',index=False)
new_df_val.to_csv('valid_contig.csv',index=False)
