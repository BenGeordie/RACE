import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
import pdb
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training params
    parser.add_argument("--data", action="store", default='criteo', type=str, help='data train size')
    parser.add_argument("--auc", action="store", default=0.795, type=float)
    parser.add_argument("--mode", action="store", default='target_auc', type=str)
                        
    args = parser.parse_args()
                        
    dataset = args.data
    mode = args.mode
    target_val_auc = args.auc
                        
    if dataset=='criteo':
        n_smpl = 33003326 
        file_path = './final_results/'
        f_base = file_path+'rslt_train_alldata_outputembed50_20220513-151927' # baseline result (train without sampling)
       # file_ls_0 = glob.glob(file_path+'rslt_end2end_train_race_outputembed50_savewght_20220523*') # sampling results 
      #  file_ls_1 = glob.glob(file_path+'rslt_end2end_train_race_outputembed50_savewght_20220525*') # sampling results
        file_ls_1 = glob.glob(file_path+'rslt_end2end_train_race_outputembed50_savewght_pretrained2epoch_2022052*')
        file_ls_2 = glob.glob(file_path+'rslt_end2end_train_random*')
#        file_ls_3 = glob.glob(file_path+'rslt_end2end_train_logloss*')
       # file_ls = file_ls_0 + file_ls_1 + file_ls_2
        file_ls = file_ls_1 + file_ls_2
        val_metric_base = pd.read_csv(f_base+'/val_metrics_final.csv')


    all_cols = ['tot_itr', 'val_loss', 'val_auc', 'val_binary_crossentropy',
           'val_binary_accuracy', 'test_loss', 'test_auc',
           'test_binary_crossentropy', 'test_binary_accuracy', 'train_time',
           'val_time', 'test_time', 'run_time','lr','repetitions','concatenations','buckets','p','num_bits','batch_size','accept_first_n','score_threshold','accept_prob', 'nnd1', 'nnd2','nnd3', 'nnd4','down_sampling_rate','final_train_time','final_run_time','total_time','w_morethan_1', 'w_1', 'train_size','w_0','hash_func', 'pre_trained']

    for col in all_cols:
        if col not in val_metric_base.columns:
            val_metric_base[col]=np.nan



    if mode=='max_val_auc':
        max_row_base = val_metric_base[val_metric_base.val_auc==max(val_metric_base.val_auc)]
        min_val_loss_base = max_row_base['val_loss'].values[0]
        max_row_val_test_auc_base = max_row_base['test_auc'].values[0]
        max_row_val_test_loss_base = max_row_base['test_loss'].values[0]

    elif mode=='target_auc':
        target_ind = val_metric_base[val_metric_base.val_auc>=target_val_auc].index.min()
        max_row_base = val_metric_base.iloc[[target_ind],:]
       # max_row_base = val_metric_base[val_metric_base.val_auc==max(val_metric_base.val_auc)]
        min_val_loss_base = max_row_base['val_loss'].values[0]
        max_row_val_test_auc_base = max_row_base['test_auc'].values[0]
        max_row_val_test_loss_base = max_row_base['test_loss'].values[0]


    df = pd.DataFrame()

    cols_1 = ['tot_itr', 'val_loss', 'val_auc', 'val_binary_crossentropy',
           'val_binary_accuracy', 'test_loss', 'test_auc',
           'test_binary_crossentropy', 'test_binary_accuracy', 'train_time',
           'val_time', 'test_time', 'run_time']
    cols_2 = ['lr','repetitions','concatenations','buckets','p','batch_size','accept_first_n','score_threshold','accept_prob', 'nnd1', 'nnd2','nnd3', 'nnd4','final_train_time','final_run_time','total_time','w_morethan_1', 'w_1', 'train_size']
    cols_3 = ['lr','repetitions','concatenations','buckets','p','batch_size','accept_first_n','score_threshold','accept_prob', 'nnd1', 'nnd2','nnd3', 'nnd4']
    cols_4 = ['tot_itr', 'val_loss', 'val_auc', 'val_binary_crossentropy',
           'val_binary_accuracy', 'test_loss', 'test_auc',
           'test_binary_crossentropy', 'test_binary_accuracy', 'train_time',
           'val_time', 'test_time', 'run_time','lr','batch_size','nnd1', 'nnd2','nnd3', 'nnd4','final_train_time','final_run_time']

    df = df.append(max_row_base[cols_4]).reset_index(drop=True, inplace=False)
    for fi in file_ls:
        if len(glob.glob(fi+'/*.csv'))>1: # if run is finished, there are two csv files in the folder
            val_metric = pd.read_csv(fi+'/val_metrics_final.csv')
            w_0 = (n_smpl - (val_metric['w_1'].values[0]+val_metric['w_morethan_1'].values[0]))/n_smpl
        elif  len(glob.glob(fi+'/*.csv'))==1: # if still running 
            val_metric = pd.read_csv(glob.glob(fi+'/*.csv')[0])
        else:
            continue

        if 'test_auc_1' in val_metric.columns:
            val_metric.rename(columns={'test_auc_1': 'test_auc'}, inplace=True)
        if 'val_auc_1' in val_metric.columns:
            val_metric.rename(columns={'val_auc_1': 'val_auc'}, inplace=True)


        for col in all_cols:
            if col not in val_metric.columns:
                val_metric[col]=np.nan

        if mode=='max_val_auc':
            max_row = val_metric[val_metric.val_auc==max(val_metric.val_auc)]
            #pdb.set_trace()
            min_val_loss = max_row['val_loss'].values[0]
            max_row_val_test_auc = max_row['test_auc'].values[0]
            max_row_val_test_loss = max_row['test_loss'].values[0]
        elif mode=='target_auc':
            #pdb.set_trace()
            target_ind = val_metric[val_metric.val_auc>=target_val_auc].index.min()
            if np.isnan(target_ind):
                max_row = val_metric.loc[[0],:]
                max_row[cols_1]=np.nan
            else:
                max_row = val_metric.iloc[[target_ind],:]
            #max_row = val_metric[val_metric.val_auc==max(val_metric.val_auc)]
            min_val_loss = max_row['val_loss'].values[0]
            max_row_val_test_auc = max_row['test_auc'].values[0]
            max_row_val_test_loss = max_row['test_loss'].values[0]

        if len(glob.glob(fi+'/*.csv'))>1:
            df_row = pd.concat([max_row[cols_1].reset_index(drop=True, inplace=False),val_metric.iloc[[0],:][cols_2+['down_sampling_rate','num_bits','hash_func','pre_trained']].reset_index(drop=True, inplace=False)],axis=1)

        else:
            df_row = pd.concat([max_row[cols_1].reset_index(drop=True, inplace=False),val_metric.iloc[[0],:][cols_3+['down_sampling_rate','num_bits','hash_func','pre_trained']].reset_index(drop=True, inplace=False)],axis=1)

        df = df.append(df_row).reset_index(drop=True, inplace=False)

    cols_to_show = ['tot_itr', 'val_loss', 'val_auc', 'val_binary_crossentropy',
            'test_loss', 'test_auc',
           'test_binary_crossentropy', 'train_time',
           'val_time', 'test_time', 'run_time','score_threshold','accept_prob','p','num_bits', 'down_sampling_rate','w_morethan_1', 'w_1', 'train_size','w_0','hash_func','pre_trained']

    tr_size = df['train_size'].values # size of subsampled data
    df['w_0'] = (n_smpl - tr_size)/n_smpl
    
    if mode=='max_val_auc':
        df[cols_to_show].to_csv('aggregated_oldscore_pretrain2epoch_results_'+mode+'.csv',index=False)
    elif mode=='target_auc':
        df[cols_to_show].to_csv('aggregated_oldscore_pretrain2epoch_results_'+mode+'_target_auc_'+str(target_val_auc)+'.csv',index=False)
        
