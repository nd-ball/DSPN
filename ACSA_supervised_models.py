#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from preprocess.PreProcess import ASAP_PreProcess, TA_PreProcess, GS_PreProcess
from models import ABAE, JPAN, CNN_Gate_Aspect_Text
from capsnet import RecurrentCapsuleNetwork, CapsuleLoss
from trainer import ACSA_supervised_model_trainer
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
    asap = ASAP_PreProcess(duplicate=True)
    w2v_model, train_set, dev_set, test_set = asap.get_dataset(duplicate=True)
elif data_name == 'TA':
    ta = TA_PreProcess(duplicate=True)
    w2v_model, train_set, dev_set, test_set = ta.get_dataset(duplicate=True)
elif data_name == 'GS':
    gs = GS_PreProcess(duplicate=True)
    w2v_model, train_set, dev_set, test_set = gs.get_dataset(duplicate=True)


# ### Train

# In[4]:


# n_vocab, d_embed = w2v_model.E.shape
# embedding = nn.Embedding(n_vocab, d_embed)
# embedding.weight = nn.Parameter(torch.from_numpy(w2v_model.E), requires_grad=False)
# aspect_embedding = nn.Embedding(w2v_model.T.shape[0], embedding_dim=200)
# aspect_embedding.weight = nn.Parameter(torch.from_numpy(w2v_model.T), requires_grad=True)

# model = RecurrentCapsuleNetwork(
#     embedding=embedding,
#     aspect_embedding=aspect_embedding,
#     num_layers=2,
#     bidirectional=True,
#     capsule_size=300, # maybe 200
#     dropout=0.5,
#     num_categories=3
# ).to(device)
# model


# In[4]:


model = CNN_Gate_Aspect_Text(w2v_model.E, w2v_model.T).to(device)
model


# Training

# In[5]:


epochs = 20
batch_size = 128
trainer = ACSA_supervised_model_trainer(data_name=data_name)


# In[6]:


trainer.train(model, train_set, dev_set=dev_set, device=device, epochs=epochs, batch_size=batch_size, 
              data_name=data_name, model_name='GCAE_' + str(seed))


# Testing

# In[9]:


model = CNN_Gate_Aspect_Text(w2v_model.E, w2v_model.T).to(device)
model.load_state_dict(torch.load("./model_params/"+ data_name +"_GCAE_" + str(seed) + "_15.model"))
model.eval()


# In[17]:


# model = RecurrentCapsuleNetwork(
#     embedding=embedding,
#     aspect_embedding=aspect_embedding,
#     num_layers=2,
#     bidirectional=True,
#     capsule_size=300, # maybe 200
#     dropout=0.5,
#     num_categories=3
# ).to(device)
# model.load_state_dict(torch.load("./model_params/"+ data_name +"_CapsNet_"+ str(seed) + "_20.model"))
# model.eval()


# In[10]:


trainer.test(model, test_set, batch_size, device)


# In[ ]:





# In[ ]:




