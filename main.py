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
import random 

def seed_everything(seed = 21):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)


device = 'cuda' if torch.cuda.is_available else 'cpu'
today=datetime.today().strftime("%Y%m%d")
os.makedirs(f'./weights/{today}',exist_ok=True)
def weightd_auc_score(pred,y_true):
    pred=torch.sigmoid(pred).detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    l1=roc_auc_score(y_true[:,0],pred[:,0])
    l2=roc_auc_score(y_true[:,1],pred[:,1])
    l3=roc_auc_score(y_true[:,2],pred[:,2])
    l4=roc_auc_score(y_true[:,3],pred[:,3])
    return 0.7*l1+0.1*l2+0.1*l3+0.1*l4

#Train
EPOCHS=20
LR=5e-4
WEIGHT_DECAY=1e-5

train_df,test_df = data.prepare_datasets()


for i in range(8):

    model = FTTransformer.make_default(
        n_num_features=len(data.NUM_COLS),
        cat_cardinalities=(data.get_Catcol_nunique(train_df)),
        d_out=4,
    )

    optimizer = optim.AdamW(model.optimization_param_groups(),lr=LR,weight_decay=WEIGHT_DECAY)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([15.83282948,  22.17171492,   7.61366862, 174.72517483])) # 라벨별로 가중치 설정
    criterion = AsymmetricLossMultiLabel() 
    model.to(device)
    criterion.to(device)
    split_train_df,split_valid_df = data.split_validation(train_df,value=i)
    train_loader,valid_loader,test_loader = dataset.get_loaders(split_train_df,split_valid_df,test_df,batch_size=512)

    max_val_score=0
    best_model=None
    for epoch in range(EPOCHS):
        total_t_pred =[]
        total_t_y =[]
        total_v_pred =[]
        total_v_y =[]
        model.train()
        #train
        print("Training Start")
        for x_cat,x_num,y in tqdm(train_loader):
            x_cat=x_cat.to(device).long()
            x_num=x_num.to(device).float()
            y=y.to(device).float()
            pred = model(x_num,x_cat)
            
            loss=criterion(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_t_pred.append(pred)
            total_t_y.append(y)
        #Validation
        with torch.no_grad():
            model.eval()
            for x_cat,x_num,y in tqdm(valid_loader):
                x_cat=x_cat.to(device).long()
                x_num=x_num.to(device).float()
                y=y.to(device).float()
                pred = model(x_num,x_cat)
                total_v_pred.append(pred)
                total_v_y.append(y)
                v_loss = criterion(pred,y).detach()
        total_t_pred = torch.cat(total_t_pred,dim=0)
        total_t_y = torch.cat(total_t_y,dim=0)
        total_v_pred = torch.cat(total_v_pred,dim=0)
        total_v_y = torch.cat(total_v_y,dim=0)
        train_score=weightd_auc_score(total_t_pred,total_t_y)
        valid_score=weightd_auc_score(total_v_pred,total_v_y)
        if valid_score>max_val_score:
            max_val_score=valid_score
            best_model=model.state_dict()
        print(f"Epoch:{epoch}/{EPOCHS}\tTrain loss : {criterion(total_t_pred,total_t_y).item()}\tValidation loss:{criterion(total_v_pred,total_v_y).item()}")
        print(f"Epoch:{epoch}/{EPOCHS}\tTrain Score : {train_score}\tValidation Score:{valid_score}")
        
    torch.save(best_model,f"weights/{today}/ASL_best_{i+1}_{max_val_score:.4f}.ckpt")
        
        
