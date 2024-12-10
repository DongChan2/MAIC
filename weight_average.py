#%%
import torch 
from copy import deepcopy
import pandas as pd 
from glob import glob 

paths=sorted(glob('weights/*.ckpt'))

ckpt=torch.load(paths[0])
out = deepcopy(ckpt)
for i in paths[1:]:
    ckpt=torch.load(i)
    for k,v in ckpt.items():
        out[k]+=v

for k,v in out.items():
    out[k]=v/len(paths)
    
torch.save(out,"Averaged_weight.ckpt")

    

    