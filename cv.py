import  warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold, ShuffleSplit

from Preprocessing import make_tfidf_df
from one_target import Train_Predict as Onetarget_Train_Predict
from all_target import Train_Predict as Alltarget_Train_Predict

def cv():
    path='../'
    train_df = pd.read_csv(path+'train_translated.csv')
    test_df = pd.read_csv(path+'test_translated.csv')
    df = pd.concat([train_df, test_df],axis=0,ignore_index=True)
    df['text_id'] = df['id']
    df.drop(columns=['id'], inplace=True)
    k = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    epoch_num=45
    
    languages = ['description','translate_de', 'translate_es', 'translate_fr', 'translate_ja']
    #languages = ['description', 'translate_es','translate_ja']
    
    off_dic={}
    loss_dic={}
    for language in languages:
        off_dic[language]={}
        loss_dic[language]={}
        
        train_df, test_df, mlp_feature = make_tfidf_df(df, language)
        alltarget_trnpred = Alltarget_Train_Predict(train_df, test_df, mlp_feature, hidden_layers=[300, 100], lr=0.0006)
        all_off_df, all_cv_loss, all_cv_score = alltarget_trnpred.make_off_df(epoch_num, k)
        
        preds_cols=['text_id', 'jobflag']+[col for col  in all_off_df.columns if 'p_' in col]
        off_dic[language]['all'] = all_off_df[preds_cols].rename(columns={col:f'{language}_{col}' for col in preds_cols})
        loss_dic[language]['cv_loss'] = all_cv_loss
        loss_dic[language]['cv_score'] = all_cv_score

        onetarget_trnpred = Onetarget_Train_Predict(train_df, test_df, mlp_feature, hidden_layers=[300, 100], lr=0.0006)
        one_off_data=[]
        for label in [1,2,3,4]:
            one_off_df, one_cv_loss, one_cv_score = onetarget_trnpred.make_off_df(label, epoch_num, k)
            preds_cols=['text_id', 'jobflag']+[col for col  in one_off_df.columns if 'p_' in col] 
            one_off_data.append(one_off_df[preds_cols].rename(columns={col:f'{language}_{col}' for col in preds_cols}))
            loss_dic[language][f'one_cv_loss_{label}'] = one_cv_loss
            loss_dic[language][f'one_cv_score_{label}'] = one_cv_score
        
        one_off_data = pd.concat(one_off_data, axis=1)
        off_dic[language]['one'] = one_off_data

    return off_dic, loss_dic

def main():
    path='../'
    train_df = pd.read_csv(path+'train_translated.csv')
    test_df = pd.read_csv(path+'test_translated.csv')
    df = pd.concat([train_df, test_df],axis=0,ignore_index=True)
    df['text_id'] = df['id']
    df.drop(columns=['id'], inplace=True)
    epoch_num=45
    
    languages = ['description','translate_de', 'translate_es', 'translate_fr', 'translate_ja']
    #languages = ['description', 'translate_de','translate_ja']
    
    test_dic={}
    loss_dic={}
    for language in languages:
        test_dic[language]={}
        loss_dic[language]={}
        
        train_df, test_df, mlp_feature = make_tfidf_df(df, language)
        alltarget_trnpred = Alltarget_Train_Predict(train_df, test_df, mlp_feature, hidden_layers=[300, 100], lr=0.0006)
        all_test_df, all_trn_loss_list = alltarget_trnpred.predict_test_df(epoch_num)
        
        preds_cols =['text_id']+[col for col  in all_test_df.columns if 'p_' in col] 
        test_dic[language]['all'] = all_test_df[preds_cols].rename(columns={col:f'{language}_{col}' for col in preds_cols})
        loss_dic[language]['all'] = all_trn_loss_list

        onetarget_trnpred = Onetarget_Train_Predict(train_df, test_df, mlp_feature, hidden_layers=[300, 100], lr=0.0006)
        one_test_data=[]
        for label in [1,2,3,4]:
            one_test_df, one_trn_loss_list = onetarget_trnpred.predict_test_df(label, epoch_num)
            preds_cols = ['text_id']+[col for col  in one_test_df.columns if 'p_' in col] 
            one_test_data.append(one_test_df[preds_cols].rename(columns={col:f'{language}_{col}' for col in preds_cols}))
            loss_dic[language][f'{label}'] = one_trn_loss_list
        one_test_data = pd.concat(one_test_data, axis=1)
        test_dic[language]['one'] = one_test_data
    
    return test_dic, loss_dic