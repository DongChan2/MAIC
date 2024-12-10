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
from glob import glob 
device = 'cuda' if torch.cuda.is_available else 'cpu'

train_df,test_df = data.prepare_datasets()

submit = pd.read_csv("Submission_example.csv")

model = FTTransformer.make_default(
    n_num_features=len(data.NUM_COLS),
    cat_cardinalities=(data.get_Catcol_nunique(train_df)),
    d_out=4,
)


weight_paths = sorted(glob(f"weights/previous/*.ckpt"))
total_pred=[]
total_test_pred=[]
for i in range(8):
    split_train_df,split_valid_df = data.split_validation(train_df,value=i)
    model.load_state_dict(torch.load(weight_paths[i]))
    model.to(device)

    train_loader,valid_loader,test_loader = dataset.get_loaders(split_train_df,split_valid_df,test_df,batch_size=512)


    preds=[]
    test_preds=[]
    #Testing
    with torch.no_grad():
        model.eval()
        #Test Stacking
        for x_cat,x_num in tqdm(test_loader):
            x_cat=x_cat.to(device).long()
            x_num=x_num.to(device).float()
            pred = model(x_num,x_cat)
            test_preds.append(pred)
        test_preds = torch.cat(test_preds,dim=0)
        test_preds=torch.sigmoid(test_preds).detach().cpu()
        total_test_pred.append(test_preds)
        #Validation Stacking
        for x_cat,x_num,y in tqdm(valid_loader):
            x_cat=x_cat.to(device).long()
            x_num=x_num.to(device).float()
            y=y.to(device).float()
            pred = model(x_num,x_cat)
            preds.append(pred)
        preds=torch.cat(preds,dim=0)
        preds=torch.sigmoid(preds).detach().cpu()
        total_pred.append(preds)
        

total_test_pred=torch.stack(total_test_pred,dim=2) # 20000x4x8
total_test_pred=total_test_pred.mean(dim=2)
test_sub=pd.DataFrame(total_test_pred,columns=submit.columns)
test_sub.to_csv(f"test_stack_previous.csv",index=False)

total_pred = torch.cat(total_pred,dim=0)
sub = pd.DataFrame(total_pred,columns=submit.columns)
sub.to_csv(f"train_stack_previous.csv",index=False)

    
    
