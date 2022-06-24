
import torch
import numpy as np
import pandas as pd
from preprocess.PreProcess import TA_PreProcess
from models import DSPN_BERT
from trainer import DSPN_BERT_trainer
import warnings
warnings.filterwarnings("ignore")


data_name = 'TA' # or 'ASAP'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True


ta = TA_PreProcess(bert=True)
w2v_model, train_set, dev_set, test_set = ta.get_dataset_bert()


# Train
model = DSPN_BERT(w2v_model.E, w2v_model.T).to(device)
model


epochs = 20
batch_size = 64
negsize = 20

trainer = DSPN_BERT_trainer()


trainer.train(model, w2v_model, train_set, dev_set, device=device, epochs=epochs, batch_size=batch_size, negsize=negsize, 
              ortho_reg=0.1, data_name=data_name, model_name='DSPN_BERT_' + str(seed))


# Test
model = DSPN_BERT(w2v_model.E, w2v_model.T).to(device)
model.load_state_dict(torch.load("./model_params/" + data_name + "_DSPN_BERT_"+ str(seed) +"_17.model", map_location=device))
model.eval()


# review-level sentiment classification
trainer.test_review_level_SC(model, test_set, batch_size, device)


# aspect identification
trainer.test_ACD(model, test_set, batch_size, device)


# ACSA
trainer.test_ACSA(model, test_set, batch_size, device, best_th=0.01918)




