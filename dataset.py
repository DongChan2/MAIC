from torch.utils.data import Dataset,DataLoader
import data 
import torch 

class Train_DS(Dataset):
    def __init__(self,dataframe):
        super().__init__()
        
        self.x_cat= torch.from_numpy(dataframe[data.CAT_COLS].values)
        self.x_num= torch.from_numpy(dataframe[data.NUM_COLS].values)
        self.y=torch.from_numpy(dataframe.iloc[:,-5:-1].values)
    def __len__(self):
        return len(self.x_cat)
    def __getitem__(self,idx):
        return self.x_cat[idx],self.x_num[idx],self.y[idx]


class Test_DS(Dataset):
    def __init__(self,dataframe):
        super().__init__()
        
        self.x_cat= torch.from_numpy(dataframe[data.CAT_COLS].values)
        self.x_num= torch.from_numpy(dataframe[data.NUM_COLS].values)

    def __len__(self):
        return len(self.x_cat)
    def __getitem__(self,idx):
        return self.x_cat[idx],self.x_num[idx]
    
    
class Train_NUMPY:
    def __init__(self,dataframe):
        self.x= (dataframe[data.CAT_COLS+data.NUM_COLS].values)
        self.y=(dataframe.iloc[:,-5:-1].values)
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    
    
class Test_NUMPY:
    def __init__(self,dataframe):
        self.x= (dataframe[data.CAT_COLS+data.NUM_COLS].values)
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx]
        
def get_loaders(dataframe,valid_dataframe,test_dataframe,batch_size=1024):
    train_ds=Train_DS(dataframe)
    valid_ds=Train_DS(valid_dataframe)
    test_ds=Test_DS(test_dataframe)
    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,pin_memory=True)
    valid_loader = DataLoader(valid_ds,batch_size=batch_size,shuffle=False,pin_memory=True)
    test_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=False,pin_memory=True)
    return train_loader,valid_loader,test_loader


def get_numpy(dataframe,valid_dataframe,test_dataframe):
    train_ds=Train_NUMPY(dataframe)
    valid_ds=Train_NUMPY(valid_dataframe)
    test_ds=Test_NUMPY(test_dataframe)

    return train_ds.x,train_ds.y,valid_ds.x,valid_ds.y,test_ds.x


