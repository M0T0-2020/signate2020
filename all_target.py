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
        self.model_trainer = Model_Trainer(output_size=4, feature=self.feature, hidden_layers=self.hidden_layers,
                                           lr=self.lr, weight_decay=self.weight_decay)
        
    def make_off_df(self, epoch_num, k):
        trn_cv_loss = []
        val_cv_loss = []
        trn_score = []
        val_score = []
        off_df=[]

        for trn, val in tqdm( k.split(self.train_df, self.train_df.jobflag), total=k.n_splits ):
            trn_df = self.train_df.iloc[trn,:]
            val_df = self.train_df.iloc[val,:]

            trn_X, trn_y = trn_df[self.feature].values.tolist(),trn_df[['jobflag']].values.tolist() 
            val_X, val_y = val_df[self.feature].values.tolist(),val_df[['jobflag']].values.tolist() 

            trn_data_set = CreateDataset(trn_X, trn_y)
            trn_dataloader = DataLoader(trn_data_set, shuffle=False, batch_size=512, sampler=ImbalancedDatasetSampler(trn_data_set))
            val_data_set = CreateDataset(val_X, val_y)
            val_dataloader = DataLoader(val_data_set, shuffle=False, batch_size=len(val_data_set))
            
            for e in range(epoch_num):
                for mm in range(4):
                    val_df[f'p_{mm+1}_{e}'] = 0
                
            trn_loss_list=[]
            val_loss_list=[]
            trn_score_list=[]
            val_score_list=[]
            self.init_model()
                
            for e in range(epoch_num):
                trn_loss_avg, trn_preds, trn_labels = self.model_trainer.train(trn_dataloader)
                val_loss_avg, p = self.model_trainer.eval(val_dataloader)
                    
                trn_preds = np.argmax(trn_preds, axis=1) + 1
                trn_score_list.append(metrics.f1_score(trn_labels, trn_preds, average='macro'))
                
                trn_loss_list.append(trn_loss_avg)
                val_loss_list.append(val_loss_avg)
                    
                for mm in range(4):
                    val_df[f'p_{mm+1}_{e}'] += p[:,mm]
            for e in range(epoch_num):
                p = val_df[[f'p_{mm+1}_{e}' for mm in range(4)]].values
                preds = np.argmax(p, axis=1)+1
                val_score_list.append( metrics.f1_score(val_df['jobflag'], preds, average='macro'))
                
            trn_cv_loss.append(trn_loss_list)
            val_cv_loss.append(val_loss_list)
            trn_score.append(trn_score_list)
            val_score.append(val_score_list)
            off_df.append(val_df)

        off_df = pd.concat(off_df, axis=0)
        off_df.sort_values('text_id', inplace=True)
        off_df.reset_index(drop=True, inplace=True)
        return off_df, trn_cv_loss, val_cv_loss, trn_score, val_score
    
    
    def predict_test_df(self, epoch_num):
        
        X, y = self.train_df[self.feature].values.tolist(), self.train_df[['jobflag']].values.tolist()
        trn_data_set = CreateDataset(X, y)
        trn_dataloader = DataLoader(trn_data_set, shuffle=False, batch_size=512, sampler=ImbalancedDatasetSampler(trn_data_set))

        val_X, val_y = self.test_df[self.feature].values.tolist(), None
        val_data_set = CreateDataset(val_X, val_y)
        val_dataloader = DataLoader(val_data_set, shuffle=False, batch_size=len(val_data_set))
        
        all_trn_loss = []
        test_df = self.test_df.copy()
        
        #make columns
        for e in range(epoch_num):
            for mm in range(4):
                test_df[f'p_{mm+1}_{e}'] = 0
        
        for loop in tqdm(range(3)):
            self.torch_random_state+=1
            torch.cuda.manual_seed_all(self.torch_random_state)
            
            self.init_model()
            trn_loss_list=[]
            
            for e in range(epoch_num):
                trn_loss_avg = self.model_trainer.train(trn_dataloader)
                p = self.model_trainer.predict(val_dataloader)
                for mm in range(4):
                    test_df[f'p_{mm+1}_{e}'] += p[:,mm]/3
                trn_loss_list.append(trn_loss_avg)
                
            all_trn_loss.append(trn_loss_list)
            
        trn_loss_list = np.mean(all_trn_loss, axis=0)

        return test_df, trn_loss_list