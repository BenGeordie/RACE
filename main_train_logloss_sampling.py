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
#from models import make_criteo_nn, make_criteo_embedding_model
from models_criteo_temp import make_criteo_nn, make_criteo_embedding_model
from models import make_avazu_embedding_model, make_clickthrough_nn
from lsh_functions import PStableHash
from race import Race
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training params
    parser.add_argument("--epoch", action="store", default=2, type=int)
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
    parser.add_argument("--first_n", action="store", default=0, type=int, help='accept fist n data points regardless')
    parser.add_argument("--thrsh", action="store", required=True, type=float, help='score threshold')
    parser.add_argument("--prob", action="store", required=True, type=float, help='accept probability')
    parser.add_argument("--data_path", action="store", required=True, type=str, help='data with score column')
    parser.add_argument("--eval_on_train", action="store_true", help='default is False')
    parser.add_argument("--train_eval_stop", action="store", default=40000, type=int,help='number of iterations we evaluate on train data when eval_on_train is True')


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
    data_path = args.data_path 
    eval_on_train_flg = args.eval_on_train
    train_eval_stop = args.train_eval_stop
    dataset = args.data
    # RACE hyper parameters
    repetitions = args.r
    concatenations = args.c
    buckets = args.b
    p = args.p
    # Weighting function hyper parameters
    accept_first_n = args.first_n
    score_threshold = args.thrsh
    accept_prob = args.prob
    
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    rslt_dir = './final_results/rslt_'+dataset+'_end2end_train_logloss_withtrainmetric_onfilteredta_'+timestr
    os.makedirs(rslt_dir)


    if dataset=='criteo':
        #train_ds = utils.load_csv_with_score('./filtered_data/data_withscore_20220514-173245/criteo_x1_small_score.csv')
        train_ds = utils.load_csv_with_score(data_path)
        val_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/data/valid_shuff_contig_trtsvalvocab_numoov0.csv')
        test_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/data/test_shuff_contig_trtsvalvocab_numoov0.csv')
        train_ds_foreval = utils.load_csv_with_score(data_path)
       # train_ds_foreval_2 = utils.load_csv_with_score(data_path)
       # val_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/criteo_x1_small.csv')
       # test_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/criteo_x1_small.csv')
       # train_ds_foreval = utils.load_criteo_csv('/home/sd73/DiverseNS/criteo_x1_small.csv')
        make_embedding_model = make_criteo_embedding_model
        n_samples = 33_003_326

    if dataset=='avazu':
        print('===data avazu====')
       # train_ds = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/train_contig_noid.csv')
        train_ds = utils.load_avazu_with_score(data_path)
        val_ds = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/valid_contig_noid.csv')
        test_ds = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/test_contig_noid.csv')
        #train_ds_foreval = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/train_contig_noid.csv')
        train_ds_foreval = utils.load_avazu_csv_with_score(data_path)
       # train_ds_foreval_2 = utils.load_avazu_csv_with_score(data_path)

        make_embedding_model = make_avazu_embedding_model
        n_samples = 32_343_173



    train_ds_batch = train_ds.batch(batch_size)
    train_ds_batch = train_ds_batch.prefetch(2)
    batch_data_val = val_ds.batch(batch_size)
    batch_data_test = test_ds.batch(batch_size)
    batch_data_train = train_ds_foreval.batch(batch_size)

   # batch_data_train_2 = train_ds_foreval_2.batch(batch_size)


    weight_fn = utils.weight_with_logloss_score(score_threshold, accept_prob)
    filtered_weighted_train_ds = utils.weight_and_filter_withscore(train_ds_batch, weight_fn)
    
   # filtered_weighted_train_ds_foreval = utils.weight_and_filter_withscore(batch_data_train_2, weight_fn) 
    # Dimensions of neural network hidden layers.
    #nn_embedding_model = make_criteo_embedding_model()
    nn_embedding_model = make_embedding_model()
    hidden_layer_dims = [args.h]*args.n
    #nn = make_criteo_nn(nn_embedding_model, hidden_layer_dims,lr)
    nn = make_clickthrough_nn(nn_embedding_model, hidden_layer_dims, lr)

    val_df = pd.DataFrame()
    sampling_w = np.array(())

    tot_itr = 0
    t00 = datetime.now()
    for ep in range(n_epoch):
        print('Epoch # =',ep)
        # in each epoch loop over batches
       # for itr, (x,y,sc,wght) in enumerate(filtered_weighted_train_ds):
        for itr, (x,y,wght) in enumerate(filtered_weighted_train_ds):
            #pdb.set_trace()
            if ep==0:
                sampling_w = np.append(sampling_w,wght.numpy().flatten())
            t1 = datetime.now()
            _ = nn.train_on_batch(x,y,wght)
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
                if eval_on_train_flg:
                    if tot_itr<train_eval_stop:
                        ttr1 = datetime.now()
                        lst_tr = nn.evaluate(batch_data_train) # evaluate on train data
                       # lst_tr = nn.evaluate(filtered_weighted_train_ds_foreval)
                        ttr2 = datetime.now()
                        eval_on_tr_time = (ttr2-ttr1) if tot_itr==0 else eval_on_tr_time + (ttr2-ttr1)
                       # lst_tr = nn.evaluate(filtered_weighted_train_ds)
                    run_time = train_time + val_time + test_time +  eval_on_tr_time
                else:
                    eval_on_tr_time = 0 # There is a column in the dataframe for this variable, in this case we should assign zero to it
                    run_time = train_time + val_time + test_time
                
                if tot_itr==0:
                    row_vals = [tot_itr]+lst_tr+lst_val+lst_test+[train_time,val_time,test_time,eval_on_tr_time,run_time]+[lr,repetitions,concatenations,buckets,p,batch_size,accept_first_n,score_threshold,accept_prob]+hidden_layer_dims
                else:
                    row_vals = [tot_itr]+lst_tr+lst_val+lst_test+[train_time,val_time,test_time,eval_on_tr_time,run_time]

                val_df = val_df.append(pd.DataFrame(row_vals).transpose())    
                if tot_itr>0:
                    os.remove(rslt_dir+'/val_metrics_itr'+str(tot_itr-eval_step)+'.csv')
                   # os.remove(rslt_dir+'/model_weights_itr'+str(tot_itr-eval_step)+'.h5')
                val_metric_cols = ['val_'+met for met in nn.metrics_names]
                test_metric_cols = ['test_'+met for met in nn.metrics_names] 
                train_metric_cols = ['train_'+met for met in nn.metrics_names]
                header_nms = ['tot_itr']+train_metric_cols+val_metric_cols+test_metric_cols+['train_time','val_time','test_time','eval_on_tr_time','run_time','lr','repetitions','concatenations','buckets','p','batch_size','accept_first_n','score_threshold','accept_prob']+['nnd'+str(ii+1) for ii in range(args.n)]
                val_df.to_csv(rslt_dir+'/val_metrics_itr'+str(tot_itr)+'.csv',header=header_nms,index=False)
               # nn.save_weights(rslt_dir+'/model_weights_itr'+str(tot_itr)+'.h5')
                #save plots
                plt.figure(figsize=(12, 10), dpi=80)
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
                plt.subplot(2,2,4)
                plt.hist(sampling_w,bins=100)
                plt.title('race weights')
                plt.savefig(rslt_dir+'/plot.png')

            tot_itr += 1
    t0f = datetime.now()
    run_time = train_time + val_time
    print('Total run time:', str(run_time))
    print('Train time =', str(train_time))
    val_df.reset_index(drop=True, inplace=True)
    val_df['final_train_time'] = np.nan
    val_df.loc[0,'final_train_time'] = train_time
    val_df['final_run_time'] = np.nan
    val_df.loc[0,'final_run_time'] = run_time
    val_df['total_time'] = np.nan
    val_df.loc[0,'total_time'] = t0f - t00
    val_df['w_morethan_1'] = np.nan
    val_df.loc[0,'w_morethan_1'] = (sampling_w>1).sum()
    val_df['w_1'] = np.nan
    val_df.loc[0,'w_1'] = (sampling_w==1).sum()
    val_df['train_size'] = np.nan
    val_df.loc[0,'train_size'] = sampling_w.size
    val_df.to_csv(rslt_dir+'/val_metrics_final.csv',header=header_nms+['final_train_time','final_run_time','total_time','w_morethan_1','w_1','train_size'],index=False)
   # nn.save_weights(rslt_dir+'/model_weights_final.h5')
    val_df.columns = header_nms+['final_train_time','final_run_time','total_time','w_morethan_1','w_1','train_size']
    np.save(rslt_dir+'/train_race_weights.npy',sampling_w)

    # Final plots 
    plt.figure(figsize=(12, 10), dpi=80)
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
    plt.subplot(2,2,4)
    plt.hist(sampling_w,bins=100)
    plt.title('race weights')
    plt.savefig(rslt_dir+'/plot.png')
