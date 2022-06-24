#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import pandas as pd
from preprocess.PreProcess import ASAP_PreProcess, TA_PreProcess, GS_PreProcess
from models import ABAE
from trainer import ABAE_trainer
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data_name = 'GS' # ['ASAP', 'TA']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True


# In[3]:


if data_name == 'ASAP':
    asap = ASAP_PreProcess()
    w2v_model, train_set, dev_set, test_set = asap.get_dataset()
elif data_name == 'TA':
    ta = TA_PreProcess()
    w2v_model, train_set, dev_set, test_set = ta.get_dataset()
elif data_name == 'GS':
    gs = GS_PreProcess()
    w2v_model, train_set, dev_set, test_set = gs.get_dataset()


# ### Train

# In[4]:


model = ABAE(w2v_model.E, w2v_model.T).to(device)
model


# In[5]:


epochs = 20
batch_size = 128
negsize = 20
trainer = ABAE_trainer(data_name=data_name)


# In[6]:


trainer.train(model=model, w2v_model=w2v_model, train_set=train_set, dev_set=dev_set, device=device, epochs=epochs, 
              batch_size=batch_size, negsize=negsize, ortho_reg=0.1, data_name=data_name, model_name='ABAE_' + str(seed))


# ### Test

# In[6]:


model = ABAE(w2v_model.E, w2v_model.T).to(device)
model.load_state_dict(torch.load("./model_params/" + data_name + "_ABAE_" + str(seed) + "_20.model", map_location=device))
model.eval()


# In[7]:


trainer.test(model, test_set, batch_size=batch_size, device=device)


# In[ ]:





# In[ ]:




