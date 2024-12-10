import torch 
import torch.nn as nn 
import torch.optim as optim 
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from rtdl import FTTransformer
from sklearn.metrics import roc_auc_score
import os 
import data 
import dataset
import glob

weight_path = glob.glob("weights/20231211/*.ckpt")
device = 'cuda' if torch.cuda.is_available else 'cpu'
train_df,test_df = data.prepare_datasets()

submit = pd.read_csv("Submission_example.csv")

model = FTTransformer.make_default(
    n_num_features=len(data.NUM_COLS),
    cat_cardinalities=(data.get_Catcol_nunique(train_df)),
    d_out=4,
)
model.load_state_dict(torch.load(weight_path))
model.to(device)


train_df,valid_df = data.split_validation(train_df,value=7)
train_loader,valid_loader,test_loader = dataset.get_loaders(train_df,valid_df,test_df,batch_size=512)


total_pred=[]
#Testing
with torch.no_grad():
    model.eval()
    for x_cat,x_num in tqdm(test_loader):
        x_cat=x_cat.to(device).long()
        x_num=x_num.to(device).float()
        pred = model(x_num,x_cat)
        total_pred.append(pred)

total_pred = torch.cat(total_pred,dim=0)
pred=torch.sigmoid(total_pred).detach().cpu().numpy()
sub=pd.DataFrame(pred,columns=submit.columns)
sub.to_csv(f"Averaged.csv",index=False)

    
    
