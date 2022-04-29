#  check weights
import utils
import importlib
# importlib.reload(utils)
import os
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# %matplotlib inline
import pdb
import sys
gpu_ind = 2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
#Config = tf.compat.v1.ConfigProto
#Config.gpu_options.allow_growth = True
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from models import make_criteo_nn, make_criteo_embedding_model
from lsh_functions import PStableHash
from race import Race

# RACE hyper parameters
repetitions = 100
concatenations = 1
buckets = 1000
p = 2.0
seed = 314150
batch_size = 2000


timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
rslt_dir = './scores/score_'+timestr
os.makedirs(rslt_dir)
params_df = pd.DataFrame()

#train_ds = utils.load_csv('/home/sd73/DiverseNS/criteo_x1_1e6.csv',39)
train_ds = utils.load_csv('/home/sd73/DiverseNS/data/train.csv',39)
train_ds_batch = train_ds.batch(batch_size)
train_ds_batch = train_ds_batch.prefetch(5)
# Weighting function hyper parameters
accept_first_n = 10000
score_threshold = 0.005
accept_prob = 0.4
race_embedding_model = make_criteo_embedding_model()
hash_module = PStableHash(race_embedding_model.output_shape[1], num_hashes=repetitions * concatenations, p=p, seed=seed)
race = Race(repetitions, concatenations, buckets, hash_module)

weight_fn = utils.weight_with_race(race, race_embedding_model, accept_first_n, score_threshold, accept_prob)

filtered_weighted_train_ds = utils.weight_and_filter(train_ds_batch, weight_fn)
#pdb.set_trace()
tr_race_w = np.array(())
race_scores = np.array(())
 
            
header_nms =['repetitions','concatenations','buckets','p','accept_first_n', 'score_threshold','accept_prob']
row_vals = [repetitions,concatenations,buckets,p,accept_first_n,score_threshold,accept_prob]
params_df = params_df.append(pd.DataFrame(row_vals).transpose())    
params_df.to_csv(rslt_dir+'/params.csv',header=header_nms,index=False)

# shuffle and make batchwise data
t0 = datetime.now()
for itr, (x,y,w) in enumerate(filtered_weighted_train_ds): 
    print('**itr**',itr)
    t1 = datetime.now() 
    tr_race_w = np.append(tr_race_w,w.numpy().flatten())
    race_scores_tnsr = race.score(race_embedding_model(x))
    race_scores = np.append(race_scores,race_scores_tnsr.numpy().flatten())
    t2 = datetime.now()
    print('**t2-t1**',t2-t1)
    sys.stdout.flush()    

    if itr%10==0:
        np.save(rslt_dir+'/race_weights.npy',tr_race_w)
        np.save(rslt_dir+'/race_scores.npy',race_scores)

print('****TOTAL TIME****',datetime.now()-t0)


np.save(rslt_dir+'/race_weights.npy',tr_race_w)
np.save(rslt_dir+'/race_scores.npy',race_scores)
plt.figure(figsize=(12, 6), dpi=80)
plt.subplot(1,2,1)
plt.hist(race_scores,bins=100)
plt.title('race scores')
plt.subplot(1,2,2)
plt.hist(tr_race_w,bins=100)
plt.title('race weights')
plt.savefig(rslt_dir+'/plot.png')
