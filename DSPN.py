#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import pandas as pd
from preprocess.PreProcess import ASAP_PreProcess, TA_PreProcess, GS_PreProcess
from models import ABAE, JPAN
from trainer import JPAN_trainer
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data_name = 'TA' # or 'ASAP'
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


model = JPAN(w2v_model.E, w2v_model.T).to(device)
model


# #### Training

# In[5]:


epochs = 20
batch_size = 128
negsize = 20

trainer = JPAN_trainer(data_name=data_name)


# In[6]:


trainer.train(model, w2v_model, train_set, dev_set, device=device, epochs=epochs, batch_size=batch_size, negsize=negsize, 
              ortho_reg=0.1, data_name=data_name, model_name='JPAN_test_' + str(seed))


# #### Testing

# In[7]:


model = JPAN(w2v_model.E, w2v_model.T).to(device)
model.load_state_dict(torch.load("./model_params/" + data_name + "_JPAN_test_"+ str(seed) +"_8.model", map_location=device))
model.eval()


# review-level sentiment classification

# In[8]:


trainer.test_review_level_SC(model, test_set, batch_size, device)


# aspect identification

# In[8]:


trainer.test_ACD(model, test_set, batch_size, device)


# ACSA

# In[7]:


trainer.test_ACSA(model, test_set, batch_size, device, best_th=0.01735)


# In[ ]:





# For analysis and error analysis

# In[8]:


y, r_senti, ac_gold, ac_pred, w_senti, word_att, p_t, flag1, flag2 = trainer.output_attention(model, test_set, device, best_th=0.01551)


# In[22]:


# AC right + review-level right
a = []
for i in range(len(flag1)):
    if flag1[i] == flag2[i] == 1:
        print(i)
        a.append(i)


# In[30]:


# AC right + review-level wrong
a = []
for i in range(len(flag1)):
    if flag1[i]==1 and flag1[i]!=flag2[i]:
        print(i)
        a.append(i)


# In[ ]:





# In[31]:


for i in a:
    # num of sentiment polarities
    if len(set([j[1] for j in ac_gold[i]])) > 1:
        print(i)
    
#     if len(ac_gold[i]) == 4:
#         print(i)


# In[ ]:





# In[ ]:


for i in range(len(flag1)):
    if len(set([j[1] for j in ac_gold[i]])) > 1:
        print(i)


# In[ ]:





# In[39]:


ind = 140
ac_gold[ind], ac_pred[ind]


# In[35]:


y[ind], r_senti[ind]


# In[36]:


w_senti.argmax(dim=-1)[ind]


# In[16]:


torch.softmax(r_senti[ind], dim=-1)


# In[13]:


cluster_map = {0: 'stuff', 1: 'value', 2: 'room', 3: 'clean', 4: 'location', 5: 'service',
                6: 'service', 7: 'room', 8: 'location', 9: 'None', 10: 'service', 11: 'room',
                12: 'location', 13: 'service', 14: 'value', 15: 'business', 16: 'None', 17: 'location',
                18: 'service', 19: 'None'}
dic = {'value': 0, 'room': 0, 'location': 0, 'clean': 0, 'stuff': 0, 'service': 0, 'business': 0, 'None': 0}
asp_imp = p_t[ind]

for i in range(len(asp_imp)):
    if asp_imp[i] > dic[cluster_map[i]]:
        dic[cluster_map[i]] = asp_imp[i]
dic.pop('None')
r = {k: v for k, v in dic.items() if v >= 0.02}
r


# In[15]:


torch.softmax(torch.tensor([0.875, 0.507, 0.694, 0.639, 0.448]), dim=-1)


# In[ ]:





# In[ ]:





# In[ ]:


word_att[ind]


# In[18]:


p_t[ind]


# In[ ]:





# In[ ]:




