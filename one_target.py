import  warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold, ShuffleSplit
from sklearn import metrics

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torchvision
from torch import optim
from torch import cuda
from torch.optim.lr_scheduler import CosineAnnealingLR

from MLP_Model import Model_Trainer
from DataLoader import CreateDataset, ImbalancedDatasetSampler

from tqdm import tqdm_notebook as tqdm

class Train_Predict:
    
    def __init__(self, train_df, test_df, feature, hidden_layers, lr=6e-4, weight_decay=0):
        self.train_df = train_df
        self.test_df = test_df
        self.feature = feature
        self.hidden_layers = hidden_layers
        self.lr=lr
        self.weight_decay = weight_decay
        
        self.torch_random_state = 2020
        torch.cuda.manual_seed_all(self.torch_random_state)
    
    def init_model(self):
        self.model_trainer = Model_Trainer(output_size=2, feature=self.feature, hidden_layers=self.hidden_layers,
                                           lr=self.lr, weight_decay=self.weight_decay)
        
    def make_off_df(self, _label, epoch_num, k):
        self.train_df['jobflag_2'] = self.train_df['jobflag'].apply(lambda x: 1 if x==_label else 0)
        
        cv_loss = []
        cv_score = []
        off_df=[]

        for trn, val in tqdm( k.split(self.train_df, self.train_df.jobflag), total=k.n_splits ):
            trn_df = self.train_df.iloc[trn,:]
            val_df = self.train_df.iloc[val,:]

            trn_X, trn_y = trn_df[self.feature].values.tolist(),trn_df[['jobflag_2']].values.tolist() 
            val_X, val_y = val_df[self.feature].values.tolist(),val_df[['jobflag_2']].values.tolist() 

            trn_data_set = CreateDataset(trn_X, trn_y)
            trn_dataloader = DataLoader(trn_data_set, shuffle=False, batch_size=256, sampler=ImbalancedDatasetSampler(trn_data_set))
            val_data_set = CreateDataset(val_X, val_y)
            val_dataloader = DataLoader(val_data_set, shuffle=False, batch_size=len(val_data_set))

            
            for e in range(epoch_num):
                val_df[f'p_{_label}_{e}'] = 0
                
            trn_loss_list=[]
            val_loss_list=[]
            trn_score_list=[]
            val_score_list=[]
            self.init_model()
                
            for e in range(epoch_num):
                trn_loss_avg, trn_preds, trn_labels = self.model_trainer.train(trn_dataloader)
                val_loss_avg, p = self.model_trainer.eval(val_dataloader)
                
                trn_preds = np.round(trn_preds[:,1])
                trn_score_list.append(metrics.accuracy_score(trn_labels, trn_preds))
                
                trn_loss_list.append(trn_loss_avg)
                val_loss_list.append(val_loss_avg)
                    
                val_df[f'p_{_label}_{e}'] += p[:,1]
            for e in range(epoch_num):
                p = val_df[[f'p_{_label}_{e}']].values
                preds = np.round(p)
                val_score_list.append( metrics.accuracy_score(val_df['jobflag_2'], preds) )
                
            cv_loss.append([trn_loss_list, val_loss_list])
            cv_score.append([trn_score_list, val_score_list])
            off_df.append(val_df)

        off_df = pd.concat(off_df, axis=0)
        off_df.sort_values('text_id', inplace=True)
        off_df.reset_index(drop=True, inplace=True)
        return off_df, cv_loss, cv_score
    
    
    def predict_test_df(self, _label, epoch_num):
        self.train_df['jobflag_2'] = self.train_df['jobflag'].apply(lambda x: 1 if x==_label else 0)
        
        X, y = self.train_df[self.feature].values.tolist(), self.train_df[['jobflag_2']].values.tolist()
        trn_data_set = CreateDataset(X, y)
        trn_dataloader = DataLoader(trn_data_set, shuffle=False, batch_size=256, sampler=ImbalancedDatasetSampler(trn_data_set))

        val_X, val_y = self.test_df[self.feature].values.tolist(), None
        val_data_set = CreateDataset(val_X, val_y)
        val_dataloader = DataLoader(val_data_set, shuffle=False, batch_size=len(val_data_set))
        
        all_trn_loss = []
        test_df = self.test_df.copy()
        
        #make columns
        for e in range(epoch_num):
            test_df[f'p_{_label}_{e}'] = 0
        
        for loop in tqdm(range(3)):
            torch_random_state=self.torch_random_state+loop
            torch.cuda.manual_seed_all(torch_random_state)
            
            self.init_model()
            trn_loss_list=[]
            
            for e in range(epoch_num):
                trn_loss_avg, _1, _2 = self.model_trainer.train(trn_dataloader)
                p = self.model_trainer.predict(val_dataloader)
                test_df[f'p_{_label}_{e}'] += p[:,1]/3
                trn_loss_list.append(trn_loss_avg)
                
            all_trn_loss.append(trn_loss_list)
            
        trn_loss_list = np.mean(all_trn_loss, axis=0)

        return test_df, trn_loss_list