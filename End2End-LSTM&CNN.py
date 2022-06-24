#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import pandas as pd
from preprocess.PreProcess import ASAP_PreProcess, TA_PreProcess, GS_PreProcess
from models import E2E_CNN, E2E_LSTM
from trainer import End2end_trainer
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data_name = 'GS' # or 'ASAP'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True


# In[3]:


if data_name == 'ASAP':
    asap = ASAP_PreProcess(end2end=True)
    w2v_model, train_set, dev_set, test_set = asap.get_dataset()
elif data_name == 'TA':
    ta = TA_PreProcess(end2end=True)
    w2v_model, train_set, dev_set, test_set = ta.get_dataset()
elif data_name == 'GS':
    gs = GS_PreProcess(end2end=True)
    w2v_model, train_set, dev_set, test_set = gs.get_dataset()


# ### Train

# In[4]:


# model = E2E_CNN(w2v_model.E, data_name=data_name).to(device)
model = E2E_LSTM(w2v_model.E, data_name=data_name).to(device)
model


# ### Train

# In[5]:


epochs = 20
batch_size = 128
trainer = End2end_trainer(data_name=data_name)


# In[6]:


trainer.train(model, train_set, dev_set, device=device, epochs=epochs, batch_size=batch_size, 
              data_name=data_name, model_name='End2end_LSTM_' + str(seed))


# Testing

# In[7]:


model = E2E_LSTM(w2v_model.E, data_name=data_name).to(device)
# model = E2E_CNN(w2v_model.E, data_name=data_name).to(device)
model.load_state_dict(torch.load("./model_params/" + data_name + "_End2end_LSTM_"+ str(seed) +"_14.model"))
model.eval()


# In[8]:


trainer.test(model, test_set, batch_size, device)


# In[ ]:





# In[ ]:




