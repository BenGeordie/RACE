# train on whole data without sampling 
import pdb
import sys
import argparse
import utils
import importlib
import os
from datetime import datetime 
from models_criteo_temp import make_criteo_nn, make_criteo_embedding_model
from models import make_avazu_embedding_model, make_clickthrough_nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", action="store", default=2, type=int)
    parser.add_argument("--batch_size", action="store", default=512, type=int)
    parser.add_argument("--eval_step", action="store", default=5000, type=int)
    parser.add_argument("--lr", action="store", default=0.001, type=float)
    parser.add_argument("--seed", action="store", default=314150, type=int)
    parser.add_argument("--h", action="store", default=800, type=int, help='hidden layer size')
    parser.add_argument("--n", action="store", default=4, type=int, help='number of hidden layers')
    parser.add_argument("--gpu", action="store", required=True, type=int)
    parser.add_argument("--data", action="store", default = 'criteo', type=str)
    parser.add_argument("--eval_on_train", action="store_true", help='default is False')
    parser.add_argument("--train_eval_stop", action="store", default=70000, type=int,help='number of iterations we evaluate on train data when eval_on_train is True')   
    parser.add_argument("--pre_train", action="store_true", help='default is False. if True,load lightly pre-trained embedding model for race')

    args = parser.parse_args()

    gpu_ind = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
    if gpu_ind != -1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    dataset = args.data
    lr = args.lr
    seed = args.seed
    n_epoch = args.epoch
    batch_size = args.batch_size
    eval_step = args.eval_step
    eval_on_train_flg = args.eval_on_train
    train_eval_stop = args.train_eval_stop
    pre_train_flg = args.pre_train

    # batch wise train and evaluation
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    #rslt_dir = './final_results/rslt_'+dataset+'_train_alldata_outputembed50_withtrainmetric_'+timestr
   # rslt_dir = './final_results/rslts03102022/rslt_'+dataset+'_train_alldata_outputembed50_'+str(n_epoch)+'epoch_'+timestr
   # rslt_dir = './final_results/rslts03102022/rslt_'+dataset+'_train_alldata_outputembed50_pretrain_itr40000_'+timestr
    rslt_dir = './final_results/rslts03102022/rslt_'+dataset+'_train_alldata_outputembed50_Nopretrain_1epoch_'+timestr
    os.makedirs(rslt_dir)
    
    if dataset=='criteo':
        train_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/data/train_shuff_contig_trtsvalvocab_numoov0.csv')
        val_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/data/valid_shuff_contig_trtsvalvocab_numoov0.csv')
        test_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/data/test_shuff_contig_trtsvalvocab_numoov0.csv')
        train_ds_foreval = utils.load_criteo_csv('/home/sd73/DiverseNS/data/train_shuff_contig_trtsvalvocab_numoov0.csv')
       # train_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/criteo_x1_small.csv')
       # val_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/criteo_x1_small.csv')
       # test_ds = utils.load_criteo_csv('/home/sd73/DiverseNS/criteo_x1_small.csv')
       # train_ds_foreval = utils.load_criteo_csv('/home/sd73/DiverseNS/criteo_x1_small.csv')
        make_embedding_model = make_criteo_embedding_model
        n_samples = 33_003_326

    if dataset=='avazu':
        print('===data avazu====')
        train_ds = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/train_contig_noid.csv')
        val_ds = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/valid_contig_noid.csv')
        test_ds = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/test_contig_noid.csv')
        train_ds_foreval = utils.load_avazu_csv('/home/bg31/RACE/Avazu/data/train_contig_noid.csv')
        make_embedding_model = make_avazu_embedding_model        
        n_samples = 32_343_173

    
    if pre_train_flg and args.data=='criteo':
     #   pdb.set_trace()
        print('======using pre-trained network for race embedding======')
       # weight_dir = 'data_with_score/rslt_train_alldata_20220517-175741/'
       # weight_dir = 'final_results/rslt_train_alldata_outputembed50_20220513-151927/' # pretrained2epoch
       # weight_dir = 'final_results/rslts03102022/rslt_criteo_train_alldata_outputembed50_1epoch_20221003-125035/' # pretrained1epochNew
        #pdb.set_trace()
        weight_dir = 'final_results/rslts03102022/rslt_criteo_train_alldata_outputembed50_itr40000_20221011-013911'
        hidden_layer_dims = [args.h]*args.n
        nn = make_clickthrough_nn(make_embedding_model(), hidden_layer_dims, lr)
        weight_file = glob.glob(weight_dir+'/model_weights_*.h5')
        nn.load_weights(weight_file[0])

    
    
    if not pre_train_flg:
        print('===Not Pre Trained===')
        #nn_embedding_model = make_criteo_embedding_model()
        nn_embedding_model = make_embedding_model()
        hidden_layer_dims = [args.h]*args.n
       # nn = make_criteo_nn(nn_embedding_model, hidden_layer_dims,lr)
        nn = make_clickthrough_nn(nn_embedding_model, hidden_layer_dims, lr)
    
    
    val_df = pd.DataFrame()
    
    # shuffle and make batchwise data
    train_ds_batch = train_ds.batch(batch_size)
    batch_data_val = val_ds.batch(batch_size)
    batch_data_test = test_ds.batch(batch_size)
    batch_data_train = train_ds_foreval.batch(batch_size)
    tot_itr = 0
    for ep in range(n_epoch):
        print('Epoch # =',ep)
        # in each epoch loop over batches
        for itr, (x,y) in enumerate(train_ds_batch):
            if tot_itr>150000:
          #  if tot_itr>40000:
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
                #if tot_itr<train_eval_stop and eval_on_train_flg:
                if eval_on_train_flg:
                    if tot_itr<train_eval_stop:
                        ttr1 = datetime.now()
                        lst_tr = nn.evaluate(batch_data_train) # evaluate on train data
                        ttr2 = datetime.now()
                        eval_on_tr_time = (ttr2-ttr1) if tot_itr==0 else eval_on_tr_time + (ttr2-ttr1)
                      #  pdb.set_trace()
                    run_time = train_time + val_time + test_time +  eval_on_tr_time
                else:
                    eval_on_tr_time = 0 # There is a column in the dataframe for this variable, in this case we should assign zero to it
                    run_time = train_time + val_time + test_time

                if tot_itr==0:
                    row_vals = [tot_itr]+lst_tr+lst_val+lst_test+[train_time,val_time,test_time,eval_on_tr_time,run_time,lr]+hidden_layer_dims
                else:
                    row_vals = [tot_itr]+lst_tr+lst_val+lst_test+[train_time,val_time,test_time,eval_on_tr_time,run_time]

                val_df = val_df.append(pd.DataFrame(row_vals).transpose())    
                if tot_itr>0:
                    os.remove(rslt_dir+'/val_metrics_itr'+str(tot_itr-eval_step)+'.csv')
                    os.remove(rslt_dir+'/model_weights_itr'+str(tot_itr-eval_step)+'.h5')
                val_metric_cols = ['val_'+met for met in nn.metrics_names] 
                test_metric_cols = ['test_'+met for met in nn.metrics_names]
                train_metric_cols = ['train_'+met for met in nn.metrics_names]
                header_nms = ['tot_itr']+train_metric_cols+val_metric_cols+test_metric_cols+['train_time','val_time','test_time','eval_on_tr_time','run_time','lr']+['nnd'+str(ii+1) for ii in range(args.n)]
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

