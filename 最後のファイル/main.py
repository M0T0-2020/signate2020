import  warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold, ShuffleSplit

from Preprocessing import make_tfidf_df
from one_target import Train_Predict as Onetarget_Train_Predict
from all_target import Train_Predict as Alltarget_Train_Predict
from MLP_Model import CustomLoss, CustomLoss_2

class Flow:
    
    def __init__(self, path='../'):
        train_df = pd.read_csv(path+'train_translated.csv')
        test_df = pd.read_csv(path+'test_translated.csv')
        df = pd.concat([train_df, test_df],axis=0,ignore_index=True)
        df['text_id'] = df['id']
        self.df = df.drop(columns=['id'])

        self.epoch_num=45

    def set_states(self, k, criterion, sampling_p):
        #k = KFold(n_splits=5, random_state=2020, shuffle=True), StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
        self.k = k

        #self.criterion = CustomLoss()
        #self.criterion = CustomLoss_2(4)
        self.criterion = criterion
        
        # 1以上 　(1 一定サンプリング  1< すこし変える)
        self.sampling_p = sampling_p

    def step_1(self):
        languages = ['description','translate_ja']
        loss_log={}
        off_data=[]
        test_data=[]
        for language in languages:
            loss_log[language]={}
            train_df, test_df, mlp_feature = make_tfidf_df(self.df, language)
            alltarget_trnpred = Alltarget_Train_Predict(
                train_df, test_df, mlp_feature, self.criterion, self.sampling_p,
                 hidden_layers=[300, 100], lr=0.0006
                 )
            all_off_df, all_cv_loss, all_cv_score = alltarget_trnpred.make_off_df(self.epoch_num, self.k)
            all_test_df, all_trn_loss_list = alltarget_trnpred.predict_test_df(self.epoch_num)
            
            loss_log['all_cv_loss'] = all_cv_loss
            loss_log['all_cv_score'] = all_cv_score
            loss_log['all_trn_loss_list'] = all_trn_loss_list

            off_preds=np.zeros((len(all_off_df), 4))
            test_preds=np.zeros((len(all_test_df), 4))
            for e in range(32, 45):
                off_preds += all_off_df[[f'p_{l+1}_{e}' for l in range(4)]].values
                test_preds += all_test_df[[f'p_{l+1}_{e}' for l in range(4)]].values

            all_off_df=all_off_df[['text_id', 'jobflag']]
            all_test_df=all_test_df[['text_id']]
            
            for i in range(4):
                all_off_df[f'p_{i+1}'] = off_preds[:,i]
                all_test_df[f'p_{i+1}'] = test_preds[:,i]

            preds_cols=[col for col  in all_off_df.columns if 'p_' in col]        
            all_off_df = all_off_df[['text_id', 'jobflag']+preds_cols].rename(
                columns={col:f'{language}_{col}' for col in preds_cols}
                )
            all_test_df = all_test_df[['text_id']+preds_cols].rename(
                columns={col:f'{language}_{col}' for col in preds_cols}
                )
            off_data.append(all_off_df)
            test_data.append(all_test_df)
        
        off_data = pd.merge(off_data[0], off_data[1], on=['text_id', 'jobflag']).sort_values('text_id')
        test_data = pd.merge(test_data[0], test_data[1], on=['text_id']).sort_values('text_id')

        return off_data, test_data, loss_log

    def step_2(self, off_data, test_data):
        loss_log={}
        mlp_feature = [col for col in off_data.columns if 'p_' in col]

        alltarget_trnpred = Alltarget_Train_Predict(
            off_data, test_data, mlp_feature, self.criterion, self.sampling_p,
             hidden_layers=[16, 4], lr=0.0006
             )
        all_off_df, all_cv_loss, all_cv_score = alltarget_trnpred.make_off_df(self.epoch_num, self.k)
        all_test_df, all_trn_loss_list = alltarget_trnpred.predict_test_df(self.epoch_num)
        
        loss_log['all_cv_loss'] = all_cv_loss
        loss_log['all_cv_score'] = all_cv_score
        loss_log['all_trn_loss_list'] = all_trn_loss_list

        return all_off_df, all_test_df, loss_log



def main():
    flow = Flow()
    ks =  [
        KFold(n_splits=5, random_state=220, shuffle=True), 
        KFold(n_splits=5, random_state=2020, shuffle=True), 
        StratifiedKFold(n_splits=5, random_state=2020, shuffle=True),
        StratifiedKFold(n_splits=5, random_state=220, shuffle=True)
        ]
    off_data=[]
    test_data=[]
    loss_log=[]
    for k in ks:
        for criterion in [CustomLoss_2(4), CustomLoss()]:
            for sampling_p in [1, 1.5]:
                flow.set_states(k, criterion, sampling_p)
                off_data, test_data, loss_log_1 = flow.step_1()
                all_off_df, all_test_df, loss_log_2 = flow.step_2(off_data, test_data)
                off_data.append(all_off_df)
                test_data.append(all_test_df)
                loss_log.append(
                    [loss_log_1, loss_log_2]
                )

    return off_data, test_data, loss_log