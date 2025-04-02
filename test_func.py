import numpy as np


class test_func():
    def __init__(self, data_name):
        if data_name == 'Trip':
            self.order = ['value', 'room', 'location', 'cleanliness', 'checkin', 'service', 'business']
        elif data_name == 'ASAP':
            self.order = ['Location', 'Service', 'Price', 'Ambience', 'Food']
    
    # 这里不考虑batch
    def eval_acd(self, asp_imp, asp_labels, th):
        preds = []
        trues = []
        for i in range(len(self.order)):
            if asp_imp[i] > th:
                preds.append(self.order[i])
            if asp_labels[i] != -1:
                trues.append(self.order[i])
                
        return trues, preds
    
    
    
    def eval_acsa(self, asp_imp, asp_senti, asp_labels, best_th):
        preds = []
        preds_c = []
        trues = []
        for i in range(len(self.order)):
            if asp_imp[i] > best_th:
                preds.append((self.order[i], asp_senti[i]))
            
            if asp_labels[i] != -1:
                trues.append((self.order[i], asp_labels[i]))
                preds_c.append((self.order[i], asp_senti[i]))

        return trues, preds, preds_c