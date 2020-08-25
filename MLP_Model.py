import  warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torchvision
from torch import optim
from torch import cuda
from torch.optim.lr_scheduler import CosineAnnealingLR

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred_1 = pred[target<1.5][:,0]
        loss_1 = torch.mean((pred_1-1)**2)
        
        pred_2 = pred[(target>1.5)&(target<2.5)][:,1]
        loss_2 = torch.mean((pred_2-1)**2)
        
        pred_3 = pred[(target>2.5)&(target<3.5)][:,2]
        loss_3 = torch.mean((pred_3-1)**2)

        pred_4 = pred[target>3.5][:,3]
        loss_4 = torch.mean((pred_4-1)**2)
        
        loss = loss_1 + loss_2 + loss_3 + loss_4
        
        return loss
    
class CustomLoss_2(nn.Module):
    def __init__(self, output_size):
        super(CustomLoss_2, self).__init__()
        self.output_size = output_size
        self.mse = nn.MSELoss()
        self.bcewithlogits = nn.BCEWithLogitsLoss()
        self.device = 'cuda' if cuda.is_available() else 'cpu'

    def forward(self, pred, target):
        
        onehot_target = torch.eye(self.output_size, device=self.device)[(target-1).long()]
        loss = self.bcewithlogits(pred, onehot_target)
        return loss


class MLPClass(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_layers[1], output_size)
        )
        
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.fc(x)
        #x= self.sigmoid(x)
        x= self.softmax(x)
        return x

class Model_Trainer:
    def __init__(self, output_size, feature, hidden_layers, lr=6e-4, weight_decay=0.0):
        self.device =  'cuda' if cuda.is_available() else 'cpu'
        self.model = MLPClass(len(feature), output_size, hidden_layers).to(self.device)
        #self.criterion = CustomLoss()
        self.criterion = CustomLoss_2(output_size)
        
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=10)
        self.output_size = output_size
        self.hidden_layers = hidden_layers

    
    def load_params(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train(self, trn_dataloader):
        self.model.train()
        avg_loss=0
        all_preds = []
        all_labels = []
        for data in trn_dataloader:
            self.optimizer.zero_grad()
            x = data['input'].to(self.device)
            label = data['label'].squeeze(1)
            if len(np.unique(label))!=self.output_size:
                continue
            label = label.to(self.device)
            x = self.model(x)   
            loss = self.criterion(x, label)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            avg_loss += loss.item()/len(trn_dataloader)
            all_preds+=x.detach().cpu().tolist()
            all_labels+=label.cpu().tolist()
            
        return avg_loss, np.array(all_preds), np.array(all_labels)

    def eval(self, val_dataloader):
        self.model.eval()
        avg_loss=0
        for data in val_dataloader:
            x = data['input'].to(self.device)
            label = data['label'].to(self.device).squeeze(1)
            x = self.model(x)
            loss = self.criterion(x, label)
            avg_loss += loss.item()/len(val_dataloader)
        return avg_loss, x.detach().cpu().numpy()
    
    def predict(self, data_loader):
        self.model.eval()
        preds = []
        for data in data_loader:
            x = data['input'].to(self.device)
            preds+=self.model(x).detach().cpu().tolist()
        return np.array(preds)