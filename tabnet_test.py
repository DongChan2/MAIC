import torch 
import torch.nn as nn 
import torch.optim as optim 
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from timm.loss import AsymmetricLossMultiLabel
from rtdl import FTTransformer
from sklearn.metrics import roc_auc_score
import os 
import data 
import dataset
import loss
from datetime import datetime
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.metrics import Metric
class weighted_auc_score(Metric):
    def __init__(self):
        self._name = "weighted_auc_score"
        self._maximize = True

    def __call__(self, y_true, y_score):
        
        l1=roc_auc_score(y_true[:,0],y_score[:,0])
        l2=roc_auc_score(y_true[:,1],y_score[:,1])
        l3=roc_auc_score(y_true[:,2],y_score[:,2])
        l4=roc_auc_score(y_true[:,3],y_score[:,3])
        return 0.7*l1+0.1*l2+0.1*l3+0.1*l4


device = 'cuda' if torch.cuda.is_available else 'cpu'
today=datetime.today().strftime("%Y%m%d")
os.makedirs(f'./weights/{today}',exist_ok=True)


#Train
EPOCHS=20
LR=5e-4
WEIGHT_DECAY=1e-5

train_df,test_df = data.prepare_datasets()


# for i in range(8):

#     split_train_df,split_valid_df = data.split_validation(train_df,value=i)
#     x_train,y_train,x_valid,y_valid,x_test = dataset.get_numpy(split_train_df,split_valid_df,test_df)
    
#     model = TabNetMultiTaskClassifier(
#         cat_idxs=[i for i in range(len(data.CAT_COLS))],
#         cat_dims=data.get_Catcol_nunique(train_df),
#         optimizer_fn=torch.optim.AdamW,
#         optimizer_params=dict(lr=LR,weight_decay=WEIGHT_DECAY),
        
#         mask_type='sparsemax' # This will be overwritten if using pretrain model
#     )
    

#     model.load_model(f"./tabnet_model_test_{i}")
#     results=[]
#     preds=model.predict_proba(x_test)
#     for i in preds:
#         results.append(i[:,1:])
#     result=np.concatenate(results,axis=1)
#     sub=pd.DataFrame(result,columns=['O_AKI','O_Critical_AKI_90','O_Death_90','O_RRT_90'])
#     sub.to_csv(f"TabNet#1.csv",index=False)
    



split_train_df,split_valid_df = data.split_validation(train_df,value=1)
x_train,y_train,x_valid,y_valid,x_test = dataset.get_numpy(split_train_df,split_valid_df,test_df)

model = TabNetMultiTaskClassifier(
    cat_idxs=[i for i in range(len(data.CAT_COLS))],
    cat_dims=data.get_Catcol_nunique(train_df),
    optimizer_fn=torch.optim.AdamW,
    optimizer_params=dict(lr=LR,weight_decay=WEIGHT_DECAY),
    
    mask_type='sparsemax' # This will be overwritten if using pretrain model
)


model.load_model(f"./tabnet_model_test_1.zip")
results=[]
preds=model.predict_proba(x_test)
for i in preds:
    results.append(i[:,1:])
result=np.concatenate(results,axis=1)
sub=pd.DataFrame(result,columns=['O_AKI','O_Critical_AKI_90','O_Death_90','O_RRT_90'])
sub.to_csv(f"TabNet#2.csv",index=False)


             

  
 
        
        
        
