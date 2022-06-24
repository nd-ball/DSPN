import torch
import numpy as np
import pandas as pd
from preprocess.PreProcess import ASAP_PreProcess, TA_PreProcess
from trainer import SC_trainer
from models import LSTMATT, TextCNN, TextRCNN, BiLSTM
import warnings
warnings.filterwarnings("ignore")


model_name = 'TextCNN' # ['TextRCNN', 'TextCNN', 'BiLSTM_Att', 'BiLSTM']
data_name = 'GS' # or 'ASAP'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True


if data_name == 'ASAP':
    asap = ASAP_PreProcess()
    w2v_model, train_set, dev_set, test_set = asap.get_dataset()
elif data_name == 'TA':
    ta = TA_PreProcess()
    w2v_model, train_set, dev_set, test_set = ta.get_dataset()
elif data_name == 'GS':
    gs = TA_PreProcess()
    w2v_model, train_set, dev_set, test_set = gs.get_dataset()


# Train

if model_name == 'TextRCNN':
    model = TextRCNN(w2v_model.E).to(device)
elif model_name == 'BiLSTM_Att':
    model = LSTMATT(w2v_model.E).to(device)
elif model_name == 'TextCNN':
    model = TextCNN(w2v_model.E).to(device)
elif model_name == 'BiLSTM':
    model = BiLSTM(w2v_model.E).to(device)

model

epochs = 20
batch_size = 128
trainer = SC_trainer(data_name=data_name)

trainer.train(model=model, train_set=train_set, dev_set=dev_set, device=device, epochs=epochs, batch_size=batch_size, 
              data_name=data_name, model_name=model_name + '_' + str(seed))

# Test
if model_name == 'TextRCNN':
    model = TextRCNN(w2v_model.E).to(device)
elif model_name == 'BiLSTM_Att':
    model = LSTMATT(w2v_model.E).to(device)
elif model_name == 'TextCNN':
    model = TextCNN(w2v_model.E).to(device)
elif model_name == 'BiLSTM':
    model = BiLSTM(w2v_model.E).to(device)

model.load_state_dict(torch.load("./model_params/" + data_name + "_" + model_name + "_" + str(seed) + "_10.model", map_location=device))
model.eval()


trainer.test(model, test_set, batch_size=batch_size, device=device)




