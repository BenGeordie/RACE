import utils
import importlib
import os
from datetime import datetime 
from models import make_clickthrough_nn, make_criteo_embedding_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
gpu_ind = 2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# batch wise train and evaluation
timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
rslt_dir = './results/rslt_'+timestr
os.makedirs(rslt_dir)


train_ds = utils.load_csv('train_contig.csv',39)
val_ds = utils.load_csv('valid_contig.csv',39)

lr = 0.001
nn_embedding_model = make_criteo_embedding_model()
hidden_layer_dims = [800, 800, 800, 800]
nn = make_clickthrough_nn(nn_embedding_model, hidden_layer_dims, lr)

seed = 314150
n_epoch = 2
batch_size = 512
eval_step = 5000
val_df = pd.DataFrame()

# shuffle and make batchwise data
train_ds_batch = train_ds.batch(batch_size)
batch_data_val = val_ds.batch(batch_size)
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
            lst = nn.evaluate(batch_data_val) # evaluate on val data!
            tv2 = datetime.now()
            val_time = (tv2-tv1) if tot_itr==0 else val_time + (tv2-tv1)
            run_time = train_time + val_time
            if tot_itr==0:
                row_vals = [tot_itr]+lst+[train_time,val_time,run_time,lr]+hidden_layer_dims
            else:
                row_vals = [tot_itr]+lst+[train_time,val_time,run_time]
                
            val_df = val_df.append(pd.DataFrame(row_vals).transpose())    
            if tot_itr>0:
                os.remove(rslt_dir+'/val_metrics_itr'+str(tot_itr-eval_step)+'.csv')
                os.remove(rslt_dir+'/model_weights_itr'+str(tot_itr-eval_step)+'.h5')
            
            header_nms = ['tot_itr']+nn.metrics_names+['train_time','val_time','run_time','lr','nnd1','nnd2','nnd3','nnd4']
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
          
val_df
