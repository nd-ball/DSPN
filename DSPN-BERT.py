#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import pandas as pd
from preprocess.PreProcess import TA_PreProcess
from models import JPAN_BERT
from trainer import JPAN_BERT_trainer
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data_name = 'TA' # or 'ASAP'
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


model = JPAN_BERT(w2v_model.E, w2v_model.T).to(device)
model


# In[4]:


epochs = 20
batch_size = 64
negsize = 20

trainer = JPAN_BERT_trainer()


# In[6]:


trainer.train(model, w2v_model, train_set, dev_set, device=device, epochs=epochs, batch_size=batch_size, negsize=negsize, 
              ortho_reg=0.1, data_name=data_name, model_name='JPAN_BERT_' + str(seed))


# ### Test

# In[5]:


model = JPAN_BERT(w2v_model.E, w2v_model.T).to(device)
model.load_state_dict(torch.load("./model_params/" + data_name + "_JPAN_BERT_"+ str(seed) +"_17.model", map_location=device))
model.eval()


# review-level sentiment classification

# In[6]:


trainer.test_review_level_SC(model, test_set, batch_size, device)


# aspect identification

# In[ ]:


trainer.test_ACD(model, test_set, batch_size, device)


# ACSA

# In[10]:


trainer.test_ACSA(model, test_set, batch_size, device, best_th=0.01918)


# In[ ]:





# In[ ]:




