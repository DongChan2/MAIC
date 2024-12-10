import torch.nn as nn 
import torch
import pandas as pd 
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import roc_auc_score
from timm.loss import AsymmetricLossMultiLabel
from rtdl import FTTransformer
from sklearn.model_selection import train_test_split
train_x=pd.read_csv('train_stack.csv').values
train_y=(pd.read_csv('Train_SyntheticAKI_MAIC2023.csv')[['O_AKI','O_Critical_AKI_90','O_Death_90','O_RRT_90']]).values
test_x=pd.read_csv('test_stack_1211.csv').values

# train_x,valid_x,train_y,valid_y=train_test_split(train_x,train_y,test_size=0.2,random_state=42)

submit=pd.read_csv("Submission_example.csv")
def weightd_auc_score(pred,y_true):
    # pred=torch.sigmoid(pred).detach().cpu().numpy()
    # y_true = y_true.detach().cpu().numpy()
    l1=roc_auc_score(y_true[:,0],pred[:,0])
    l2=roc_auc_score(y_true[:,1],pred[:,1])
    l3=roc_auc_score(y_true[:,2],pred[:,2])
    l4=roc_auc_score(y_true[:,3],pred[:,3])
    return 0.7*l1+0.1*l2+0.1*l3+0.1*l4
class Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(4,2)
        
        # self.bn=nn.BatchNorm1d(2)

        self.fc=nn.Linear(2,4)

    def forward(self,x):
        x=self.input(x)
        # x=self.bn(x)
        x=self.fc(x)
        return x
class train_data(Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return torch.from_numpy(self.x[idx]),torch.from_numpy(self.y[idx])
class test_data(Dataset):
    def __init__(self,x):
        super().__init__()
        self.x=x
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return torch.from_numpy(self.x[idx])
train_ds=train_data(train_x,train_y)
test_ds=test_data(test_x)

train_loader = DataLoader(train_ds,batch_size=512,shuffle=True)
test_loader=DataLoader(test_ds,batch_size=512,shuffle=False)
model=Ensemble()

# criterion=nn.BCEWithLogitsLoss()
criterion=AsymmetricLossMultiLabel()
optimizer= torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
model.to('cuda')
criterion.to('cuda')
min_loss=999
for i in range(30):
    model.train()
    epoch_loss=0
    for x,y in train_loader:
        x=x.to('cuda').float()
        y=y.to('cuda').float()
        pred=model(x)
        loss=criterion(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    epoch_loss/=len(train_loader)
    print(f"Epoch:{i+1}\tloss:{(epoch_loss)}")
    
    if epoch_loss<min_loss:
        min_loss=epoch_loss
        torch.save(model.state_dict(),'ensemble5.ckpt')

preds=[]
with torch.no_grad():
    model.load_state_dict(torch.load("ensemble5.ckpt"))
    model.eval()
    for x in test_loader:
        x=x.to('cuda').float()
        pred=model(x)
        pred=torch.sigmoid(pred)
        preds.append(pred)
preds=torch.cat(preds,dim=0).detach().cpu().numpy()
sub=pd.DataFrame(preds,columns=submit.columns)
sub.to_csv(f"StackingEnsemble5.csv",index=False)
