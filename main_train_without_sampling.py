# train on whole data without sampling 
import pdb
import sys
import argparse
import utils
import importlib
import os
from datetime import datetime 
from models import make_clickthrough_nn, make_criteo_embedding_model, make_avazu_embedding_model, make_movielens_embedding_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", action="store", default=2, type=int)
    parser.add_argument("--batch_size", action="store", default=512, type=int)
    parser.add_argument("--eval_step", action="store", default=5000, type=int)
    parser.add_argument("--lr", action="store", default=0.001, type=float)
    parser.add_argument("--seed", action="store", default=314150, type=int)
    parser.add_argument("--h", action="store", default=1024, type=int, help='hidden layer size')
    parser.add_argument("--n", action="store", default=4, type=int, help='number of hidden layers')
    parser.add_argument("--gpu", action="store", required=True, type=int)
    
    
    parser.add_argument("--data", action="store", default = 'criteo', type=str)
    args = parser.parse_args()

    gpu_ind = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
    if gpu_ind != -1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)


    # batch wise train and evaluation
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    rslt_dir = './final_results/rslt_train_alldata_'+timestr
    os.makedirs(rslt_dir)
    
    if args.data=='criteo':
        train_ds = utils.load_criteo_csv('/DiverseNS/data/train.csv')
        val_ds = utils.load_criteo_csv('/DiverseNS/data/valid.csv')
        test_ds = utils.load_criteo_csv('/DiverseNS/data/test.csv')
        make_embedding_model = make_criteo_embedding_model
    if args.data=='avazu':
        train_ds = utils.load_avazu_csv('/Users/benitogeordie/Downloads/Avazu_x4/train_contig_noid.csv')
        val_ds = utils.load_avazu_csv('/Users/benitogeordie/Downloads/Avazu_x4/valid_contig_noid.csv')
        test_ds = utils.load_avazu_csv('/Users/benitogeordie/Downloads/Avazu_x4/test_contig_noid.csv')
        make_embedding_model = make_avazu_embedding_model
    if args.data=='movielens':
        train_ds = utils.load_movielens_csv('/Users/benitogeordie/Downloads/Movielenslatest_x1/train_contig.csv')
        val_ds = utils.load_movielens_csv('/Users/benitogeordie/Downloads/Movielenslatest_x1/valid_contig.csv')
        test_ds = utils.load_movielens_csv('/Users/benitogeordie/Downloads/Movielenslatest_x1/test_contig.csv')
        make_embedding_model = make_movielens_embedding_model


    lr = args.lr
    seed = args.seed
    n_epoch = args.epoch
    batch_size = args.batch_size
    eval_step = args.eval_step
    
    nn_embedding_model = make_embedding_model()
    hidden_layer_dims = [args.h]*args.n
    nn = make_clickthrough_nn(nn_embedding_model, hidden_layer_dims,lr)
    
    
    val_df = pd.DataFrame()
    
    # shuffle and make batchwise data
    train_ds_batch = train_ds.batch(batch_size)
    batch_data_val = val_ds.batch(batch_size)
    batch_data_test = test_ds.batch(batch_size)
    tot_itr = 0
    for ep in range(n_epoch):
        print('Epoch # =',ep)
        # in each epoch loop over batches
        for itr, (x,y) in enumerate(train_ds_batch):
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
    run_time = train_time + val_time
    print('Total run time:', str(run_time))
    print('Train time =', str(train_time))
    val_df.reset_index(drop=True, inplace=True)
    val_df['final_train_time'] = np.nan
    val_df.loc[0,'final_train_time'] = train_time
    val_df['final_run_time'] = np.nan
    val_df.loc[0,'final_run_time'] = run_time
    val_df.to_csv(rslt_dir+'/val_metrics_final.csv',header=header_nms+['final_train_time','final_run_time'],index=False)
    nn.save_weights(rslt_dir+'/model_weights_final.h5')
    val_df.columns = header_nms+['final_train_time','final_run_time']

    # Final plots 
    plt.figure(figsize=(12, 18), dpi=80)
    plt.subplot(2,2,1)
    plt.plot(val_df.values[:,0],val_df.values[:,2])
    plt.title(nn.metrics_names[1])
    plt.xlabel('iteration')
    plt.subplot(2,2,2)
    plt.plot(val_df.values[:,0],val_df.values[:,3])
    plt.title(nn.metrics_names[2])
    plt.xlabel('iteration')
    plt.subplot(2,2,3)
    plt.plot(val_df.values[:,0],val_df.values[:,4])
    plt.title(nn.metrics_names[3])
    plt.xlabel('iteration')
    plt.savefig(rslt_dir+'/plot.png')
