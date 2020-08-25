import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torchvision
from torch import optim
from torch import cuda
from torch.optim.lr_scheduler import CosineAnnealingLR

class Bert_Trainer:
    def __init__(self, output_size, criterion):
        self.device =  'cuda' if cuda.is_available() else 'cpu'
        self.model = BERTClass(0.4, output_size).to(self.device)
        self.criterion = criterion
        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=2e-5)
        self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=10)
        self.output_size = output_size

    
    def load_params(self, state_dict):
        self.model.load_state_dict(state_dict)

    
    def train(self, trn_dataloader):
        self.model.train()
        avg_loss=0
        for data in tqdm(trn_dataloader):
            self.optimizer.zero_grad()
            ids = data['ids'].to(self.device)
            mask = data['mask'].to(self.device)
            labels = data['labels'].to(self.device)
            outputs = self.model(ids, mask)
            del ids, mask; gc.collect()
            
            if self.output_size==1:
                labels = labels.squeeze(1)
                outputs = outputs.squeeze(1)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            avg_loss += loss.item()/len(trn_dataloader)
        
        return avg_loss

    def eval(self, val_dataloader):
        self.model.eval()
        avg_loss=0
        for data in tqdm(val_dataloader):
            self.optimizer.zero_grad()
            ids = data['ids'].to(self.device)
            mask = data['mask'].to(self.device)
            labels = data['labels'].to(self.device)
            outputs = self.model(ids, mask)
            del ids, mask; gc.collect()

            if self.output_size==1:
                labels = labels.squeeze(1)
                outputs = outputs.squeeze(1)
                
            loss = self.criterion(outputs, labels)
            avg_loss += loss.item()/len(val_dataloader)
        return avg_loss

    def predict(self, data_loader):
        self.model.eval()
        preds = []
        for data in tqdm(data_loader):
            self.optimizer.zero_grad()
            ids = data['ids'].to(self.device)
            mask = data['mask'].to(self.device)
            outputs=self.model(ids, mask).detach().cpu().numpy()
            del ids, mask; gc.collect()

            if self.output_size==1:
                outputs = outputs.flatten()
            else:
                outputs = outputs[:,1]
            preds+=outputs.tolist()
            
        return preds