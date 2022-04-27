import utils
import importlib
import os
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from models import make_criteo_nn, make_criteo_embedding_model
from lsh_functions import PStableHash
from race import Race

# RACE hyper parameters
repetitions = 10
concatenations = 1
buckets = 100
p = 2.0
seed = 314150
batch_size = 25


timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
rslt_dir = './scores/score_'+timestr
os.makedirs(rslt_dir)
params_df = pd.DataFrame()

train_ds = utils.load_csv('criteo_x1_small.csv',39)

# Weighting function hyper parameters
accept_first_n = 100
score_threshold = 0.0001
accept_prob = 0.1

# save params
header_nms =['repetitions','concatenations','buckets','p','accept_first_n', 'score_threshold','accept_prob']
row_vals = [repetitions,concatenations,buckets,p,accept_first_n,score_threshold,accept_prob]
params_df = params_df.append(pd.DataFrame(row_vals).transpose())    
params_df.to_csv(rslt_dir+'/params.csv',header=header_nms,index=False)

# race 
race_embedding_model = make_criteo_embedding_model()
hash_module = PStableHash(race_embedding_model.output_shape[1], num_hashes=repetitions * concatenations, p=p, seed=seed)
race = Race(repetitions, concatenations, buckets, hash_module)

weight_fn = utils.weight_with_race(race, race_embedding_model, accept_first_n, score_threshold, accept_prob)
filtered_weighted_train_ds = utils.weight_and_filter(train_ds, weight_fn)


# make batchwise data
batch_data_train = filtered_weighted_train_ds.batch(batch_size)


race_w = np.array(())
race_scores = np.array(())

t0 = datetime.now()
for itr, (x,y,w) in enumerate(batch_data_train):
    
    race_w = np.append(race_w,w.numpy().flatten())
    race_scores_tnsr = race.score(race_embedding_model(x))
    race_scores = np.append(race_scores,race_scores_tnsr.numpy().flatten())
    

    if itr%10==0:
        np.save(rslt_dir+'/race_weights.npy',race_w)
        np.save(rslt_dir+'/race_scores.npy',race_scores)
        plt.figure(figsize=(12, 6), dpi=80)
        plt.subplot(1,2,1)
        plt.hist(race_scores,bins=100)
        plt.title('race scores')
        plt.subplot(1,2,2)
        plt.hist(race_w,bins=100)
        plt.title('race weights')
        plt.savefig(rslt_dir+'/plot.png')

print('TOTAL TIME:',datetime.now()-t0)