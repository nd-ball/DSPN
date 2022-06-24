#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import pandas as pd
from preprocess.PreProcess import TA_PreProcess
from trainer import SC_BERT_trainer
from models import BERT
import warnings
warnings.filterwarnings("ignore")


# In[2]:


model_name = 'BERT'
data_name = 'TA'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True


# In[3]:


ta = TA_PreProcess(bert=True)
w2v_model, train_set, dev_set, test_set = ta.get_dataset_bert()


# ### Train

# In[4]:


model = BERT(w2v_model.E).to(device)
model


# In[4]:


epochs = 20
batch_size = 64
trainer = SC_BERT_trainer(data_name=data_name)


# In[6]:


trainer.train(model=model, train_set=train_set, dev_set=dev_set, device=device, epochs=epochs, batch_size=batch_size, 
              data_name=data_name, model_name=model_name + '_' + str(seed))


# ### Test

# In[5]:


model = BERT(w2v_model.E).to(device)
model.load_state_dict(torch.load("./model_params/" + data_name + "_" + model_name + "_" + str(seed) + "_10.model", map_location=device))
model.eval()


# In[6]:


trainer.test(model, test_set, batch_size=batch_size, device=device)


# In[ ]:





# In[ ]:




