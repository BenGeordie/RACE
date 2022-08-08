# train by loading the scores computed offline
import shutil
import os
import pdb
import sys
import argparse
import utils
import importlib
from datetime import datetime 
#from models import make_criteo_nn, make_criteo_embedding_model
from models import make_avazu_embedding_model, make_criteo_embedding_model, make_clickthrough_nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", action="store", default=1, type=int)
    parser.add_argument("--batch_size", action="store", default=512, type=int)
    parser.add_argument("--eval_step", action="store", default=5000, type=int)
    parser.add_argument("--lr", action="store", default=0.001, type=float)
    parser.add_argument("--seed", action="store", default=314150, type=int)
    parser.add_argument("--h", action="store", default=800, type=int, help='hidden layer size')
    parser.add_argument("--n", action="store", default=4, type=int, help='number of hidden layers')
    parser.add_argument("--gpu", action="store", required=True, type=int)
    parser.add_argument("--stop_itr", action="store", default=40000, type=int)
    
    
    
    parser.add_argument("--data", action="store", default = 'criteo', type=str)
    args = parser.parse_args()

    gpu_ind = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
    if gpu_ind != -1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

    dataset = args.data

    # batch wise train and evaluation
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    rslt_dir = './data_with_score/rslt_'+dataset+'_train_alldata_'+timestr
    os.makedirs(rslt_dir)

    
    if dataset=='criteo':
        print('===data criteo====')
        tr_data_path = '/home/sd73/DiverseNS/data/train_shuff_contig_trtsvalvocab_numoov0.csv'
        tr_data_path = '/home/sd73/DiverseNS/criteo_x1_small.csv'
        train_ds = utils.load_criteo_csv(tr_data_path)
        val_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/data/valid_shuff_contig_trtsvalvocab_numoov0.csv')
        test_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/data/test_shuff_contig_trtsvalvocab_numoov0.csv')
       # train_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/criteo_x1_small.csv')
        val_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/criteo_x1_small.csv')
        test_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/criteo_x1_small.csv')
        make_embedding_model = make_criteo_embedding_model 
    
    if dataset=='avazu':
        print('===data avazu====')
        tr_data_path = '/home/bg31/RACE/Avazu/data/train_contig_noid.csv'
        train_ds = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/train_contig_noid.csv')
        val_ds = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/valid_contig_noid.csv')
        test_ds = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/test_contig_noid.csv')
        #train_ds_foreval = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/train_contig_noid.csv')
        make_embedding_model = make_avazu_embedding_model
        n_samples = 32_343_173


    lr = args.lr
    seed = args.seed
    n_epoch = args.epoch
    batch_size = args.batch_size
    eval_step = args.eval_step
    stop_itr = args.stop_itr
    

    #nn_embedding_model = make_criteo_embedding_model()
    nn_embedding_model = make_embedding_model()
    hidden_layer_dims = [args.h]*args.n
    #nn = make_criteo_nn(nn_embedding_model, hidden_layer_dims, lr)
    nn = make_clickthrough_nn(nn_embedding_model, hidden_layer_dims, lr)

    val_df = pd.DataFrame()

    train_ds_batch = train_ds.batch(batch_size)
    batch_data_val = val_ds.batch(batch_size)
    batch_data_test = test_ds.batch(batch_size)
    tot_itr = 0
    for ep in range(n_epoch):
        print('Epoch # =',ep)
        # in each epoch loop over batches
        for itr, (x,y) in enumerate(train_ds_batch):
            if tot_itr> stop_itr:
                break
            t1 = datetime.now()
            _ = nn.train_on_batch(x,y)
            t2 = datetime.now()
            train_time = (t2-t1) if tot_itr==0 else train_time + (t2-t1)
            if tot_itr%eval_step==0:
                print('   Iteration # =',tot_itr)
                tv1 = datetime.now()
                lst_val = nn.evaluate(batch_data_val) # evaluate on val data
                tv2 = datetime.now()
                tt1 = datetime.now()
                lst_test = nn.evaluate(batch_data_test) # evaluate on test data
                tt2 = datetime.now()
                val_time = (tv2-tv1) if tot_itr==0 else val_time + (tv2-tv1)
                test_time = (tt2-tt1) if tot_itr==0 else test_time + (tt2-tt1)
                run_time = train_time + val_time + test_time
                if tot_itr==0:
                    row_vals = [tot_itr]+lst_val+lst_test+[train_time,val_time,test_time,run_time,lr]+hidden_layer_dims
                   
                    
                else:
                    row_vals = [tot_itr]+lst_val+lst_test+[train_time,val_time,test_time,run_time]

                val_df = val_df.append(pd.DataFrame(row_vals).transpose())    
                if tot_itr>0:
                    os.remove(rslt_dir+'/val_metrics_itr'+str(tot_itr-eval_step)+'.csv')
                    os.remove(rslt_dir+'/model_weights_itr'+str(tot_itr-eval_step)+'.h5')
                val_metric_cols = ['val_'+met for met in nn.metrics_names] 
                test_metric_cols = ['test_'+met for met in nn.metrics_names]
                header_nms = ['tot_itr']+val_metric_cols+test_metric_cols+['train_time','val_time','test_time','run_time','lr']+['nnd'+str(ii+1) for ii in range(args.n)]
                val_df.to_csv(rslt_dir+'/val_metrics_itr'+str(tot_itr)+'.csv',header=header_nms,index=False)
                nn.save_weights(rslt_dir+'/model_weights_itr'+str(tot_itr)+'.h5')

            tot_itr+=1   
            
    print('=====final itr======',tot_itr)

    
    # load pretrained network weights to compute value of logloss for each data point, as its score
    del nn
    #nn_embedding_model = make_criteo_embedding_model()
    nn_embedding_model = make_embedding_model()
    hidden_layer_dims = [args.h]*args.n
   # nn = make_criteo_nn(nn_embedding_model, hidden_layer_dims, lr)   
    nn = make_clickthrough_nn(nn_embedding_model, hidden_layer_dims, lr)
    weight_file = glob.glob(rslt_dir+'/model_weights_*.h5')
    nn.load_weights(weight_file[0])
    
    if dataset=='criteo': # reload data to reset the iterator
        train_ds = utils.load_criteo_csv(tr_data_path)
    
    if dataset=='avazu': # reload data to reset the iterator
        train_ds = utils.load_avazu_csv(tr_data_path)


    logloss_scores = np.array(())
    for itr, (x,y) in enumerate(train_ds_batch):
        p = nn.predict(x)
        y_rshp = tf.reshape(y,(-1,1))
# manual computation
#         term1 = -tf.math.multiply(tf.cast(y_rshp, dtype=tf.float32),tf.math.log(tf.constant(p)))
#         term2 = -tf.math.multiply(tf.cast(1-y_rshp, dtype=tf.float32),tf.math.log(tf.constant(1-p)))
#         tf.reshape((term2 + term1),(25,))
#         scores = term2 + term1 # logloss
#         logloss_scores = np.append(logloss_scores,scores.numpy().flatten())
        # tensorflow computation 
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, axis=1,reduction=tf.keras.losses.Reduction.NONE)
        scores = bce(y_rshp, p)
        logloss_scores = np.append(logloss_scores,scores.numpy().flatten())

    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    data_with_score_dir = './data_with_score/data_'+dataset+'_withscore_'+timestr
    os.makedirs(data_with_score_dir)
    #if dataset=='criteo':
    train_ds_score = pd.read_csv(tr_data_path)
    train_ds_score['scores']=logloss_scores# add score column
    train_ds_score.to_csv(data_with_score_dir+'/'+dataset+'_withscore_itr'+str(stop_itr)+'.csv',index=None)

