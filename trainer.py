import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader
from test_func import test_func
from pytorch_pretrained_bert import BertAdam


def max_margin_loss(r_s, z_s, z_n):
    device = r_s.device
    pos = torch.bmm(z_s.unsqueeze(1), r_s.unsqueeze(2)).squeeze(2)
    negs = torch.bmm(z_n, r_s.unsqueeze(2)).squeeze()
    J = torch.ones(negs.shape).to(device) - pos.expand(negs.shape) + negs
    return torch.sum(torch.clamp(J, min=0.0))


def orthogonal_regularization(T):
    T_n = F.normalize(T, dim=1)
    I = torch.eye(T_n.shape[0]).to(T_n.device)
    return torch.norm(T_n.mm(T_n.t()) - I)


class ABAE_trainer():
    def __init__(self, data_name):
        print("Preparing...")
        self.test_func = test_func(data_name=data_name)

    def train(self, model, train_set, dev_set, device='cuda', n_epochs=20, batch_size=32, negsize=5, ortho_reg=0.1, data_name='', model_name=''):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        opt = BertAdam(optimizer_grouped_parameters,
                             lr=3e-5,
                             warmup=0.05,
                             t_total=len(train_set) * n_epochs)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9)
        model.train()
        train_losses = []
        best_dev_loss = float('inf')
        for e in range(n_epochs):
            train_dataloader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
            for batch in tqdm.tqdm(train_dataloader):
                X = batch[0].to(device)
                # rating = batch[1].to(device)
                # aspects = batch[2].to(device)
                ##### 这里的X_neg其实是一个stack，那negsize是不是其实没有意义？
                X_neg = torch.stack(tuple([X[torch.randperm(X.shape[0])[:negsize]] for _ in range(batch_size)])).to(device)
                r_s, z_s, z_n, p_t = model(X, X_neg)
                J = max_margin_loss(r_s, z_s, z_n)
                U = orthogonal_regularization(model.T.weight)
                loss = J + ortho_reg * batch_size * U
                train_losses.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            model.eval()
            dev_loss = self.evaluate(model, dev_set, device, batch_size, negsize, ortho_reg)
            model.train()
            print("EPOCH:", e+1, "TRAIN-LOSS", np.mean(train_losses), "DEV-LOSS",dev_loss)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = e + 1
                # saving model
                torch.save(model.state_dict(), './model_params/' + str(data_name)  + '_' + str(model_name) +'.model')

            scheduler.step()

        model.eval()
        

    def evaluate(self, model, dev_set, device='cuda', batch_size=8, negsize=5, ortho_reg=0.1):
        dev_dataloader = DataLoader(dev_set, batch_size=batch_size, drop_last=True)
        dev_losses = []
        with torch.no_grad():
            for batch in dev_dataloader:
                X = batch[0].to(device)
                X_neg = torch.stack(tuple([X[torch.randperm(X.shape[0])[:negsize]] for _ in range(batch_size)])).to(device)
                
                r_s, z_s, z_n, p_t = model(X, X_neg)
                J = max_margin_loss(r_s, z_s, z_n)
                U = orthogonal_regularization(model.T.weight)
                loss = J + ortho_reg * batch_size * U
                dev_losses.append(loss.item())
                
        return np.mean(dev_losses)

    
    def test_acd(self, model, test_set, batch_size, device='cuda'):
        test_dataloader = DataLoader(test_set, batch_size=batch_size, drop_last=True)
        with torch.no_grad():
            for th in np.linspace(0, 1e-2, 20):
                S = 0
                G = 0
                S_G = 0
                for batch in test_dataloader:
                    X = batch[0].to(device)
                    aspects = batch[2].to(device)
                    _, p_t = model.get_aspect_importance(X)
                    
                    for b in range(X.shape[0]):
                        trues, preds = self.test_func.eval_acd(p_t[b].tolist(), aspects[b].tolist(), th)
                        
                        S += len(preds)
                        G += len(trues)
                        S_G += len(list(set(preds).intersection(set(trues))))

                P = S_G / S
                R = S_G / G
                F1 = (2 * P * R) / (P + R)

                print('Th: %0.8f | P: %0.8f | R: %0.8f | F1: %0.8f' % (th, P, R, F1))


                
class RP_trainer():
    def __init__(self):
        self.loss_func = nn.CrossEntropyLoss()
        self.step_size = 5

    def train(self, model, train_set, dev_set, device='cuda', n_epochs=5, batch_size=32, data_name='', model_name=''):
        model.train()
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0}]
        opt = BertAdam(optimizer_grouped_parameters,
                             lr=3e-5,
                             warmup=0.05,
                             t_total=len(train_set) * n_epochs)
        
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=self.step_size, gamma=0.9)
        train_f1 = []
        train_losses = []
        best_dev_f1 = 0
        best_epoch = 1
        for e in range(n_epochs):
            train_dataloader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
            for batch in tqdm.tqdm(train_dataloader):
                X = batch[0].to(device)
                rating = batch[1].to(device)
                y_pred = model(X)
                loss = self.loss_func(y_pred, rating)
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                trues = rating.tolist()
                preds = y_pred.argmax(dim=1).tolist()
                
                # acc = (y_pred.argmax(dim=1) == y).sum().cpu().item()/y.shape[0]
                train_f1.append(f1_score(trues, preds, average='macro'))
                train_losses.append(loss.item())
            
            model.eval()
            dev_f1 = self.evaluate(model, dev_set, batch_size, device)
            model.train()
            print("EPOCH:", e+1, "TRAIN-F1:", np.mean(train_f1), "TRAIN-LOSS", np.mean(train_losses), "DEV-F1",dev_f1)
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_epoch = e + 1
                # saving model
                torch.save(model.state_dict(), './model_params/' + str(data_name)  + '_' + str(model_name) +'.model')
            
            scheduler.step()
        model.eval()

    
    def evaluate(self, model, dev_set, batch_size, device='cuda'):
        dev_dataloader = DataLoader(dev_set, batch_size=batch_size, drop_last=True)
        dev_f1_list = []
        with torch.no_grad():
            for batch in dev_dataloader:
                X = batch[0].to(device)
                rating = batch[1].to(device)
                y_pred = model(X)
                
                trues = rating.tolist()
                preds = y_pred.argmax(dim=1).tolist()
                dev_f1_list.append(f1_score(trues, preds, average='macro'))
                
        return np.mean(dev_f1_list)

    def test_rp(self, model, test_set, batch_size, device):
        test_dataloader = DataLoader(test_set, batch_size=batch_size, drop_last=True)
        trues = []
        preds = []
        with torch.no_grad():
            for batch in test_dataloader:
                X = batch[0].to(device)
                rating = batch[1].to(device)
                y_pred = model(X)
                preds += y_pred.argmax(dim=1).tolist()
                trues += rating.tolist()

        print("Precision:", precision_score(trues, preds, average='macro'))
        print("Recall:", recall_score(trues, preds, average='macro'))
        print("F1-score:", f1_score(trues, preds, average='macro'))
        print("Accuracy:", accuracy_score(trues, preds))

        
class DSPN_trainer():
    def __init__(self, data_name):
        self.data_name = data_name
        self.loss_func = nn.CrossEntropyLoss()
        self.step_size = 5
        self.gamma = 0.9
        self.test_func = test_func(data_name=data_name)

    def train(self, model, train_set, dev_set, device='cuda', n_epochs=5, batch_size=32, negsize=5, ortho_reg=0.1, data_name='', model_name='DSPN'):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        opt = BertAdam(optimizer_grouped_parameters,
                             lr=3e-5,
                             warmup=0.05,
                             t_total=len(train_set) * n_epochs)
        
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=self.step_size, gamma=self.gamma)
        model.train()
        train_losses = []
        best_dev_loss = float('inf')
        for e in range(n_epochs):
            train_dataloader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
            for batch in tqdm.tqdm(train_dataloader):
                X = batch[0].to(device)
                rating = batch[1].to(device)
                aspects = batch[2].to(device)
                X_neg = torch.stack(tuple([X[torch.randperm(X.shape[0])[:negsize]] for _ in range(batch_size)])).to(device)
                
                r_s, z_s, z_n, p_t, r_senti_pred, a_senti_pred = model(X, X_neg)
                J = max_margin_loss(r_s, z_s, z_n)
                U = orthogonal_regularization(model.T.weight)
                loss_acd = J + ortho_reg * batch_size * U
                loss_py = self.loss_func(r_senti_pred, rating)
                loss = 0.01 * loss_acd + loss_py
                train_losses.append(loss.item())
                
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            dev_loss = self.evaluate(model, dev_set, batch_size, negsize, ortho_reg, device)
            model.train()
            print("EPOCH:", e+1, "TRAIN-LOSS:", np.mean(train_losses), "DEV-LOSS:", dev_loss)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                # Saving model
                torch.save(model.state_dict(), './model_params/' + str(data_name) + '_' + str(model_name) + '.model')
            
            scheduler.step()
        model.eval()

    
    def evaluate(self, model, dev_set, batch_size, negsize, ortho_reg, device='cuda'):
        dev_dataloader = DataLoader(dev_set, batch_size=batch_size, drop_last=True)
        dev_losses = []
        with torch.no_grad():
            for batch in dev_dataloader:
                X = batch[0].to(device)
                rating = batch[1].to(device)
                aspects = batch[2].to(device)
                X_neg = torch.stack(tuple([X[torch.randperm(X.shape[0])[:negsize]] for _ in range(batch_size)])).to(device)
                
                r_s, z_s, z_n, p_t, r_senti_pred, a_senti_pred = model(X, X_neg)
                J = max_margin_loss(r_s, z_s, z_n)
                U = orthogonal_regularization(model.T.weight)
                loss_acd = J + ortho_reg * batch_size * U
                loss_py = self.loss_func(r_senti_pred, rating)
                loss = 0.01 * loss_acd + loss_py
                dev_losses.append(loss.item())

        return np.mean(dev_losses)

    
    def test_rp(self, model, test_set, batch_size, device):
        test_dataloader = DataLoader(test_set, batch_size=batch_size, drop_last=True)
        trues = []
        preds = []
        with torch.no_grad():
            for batch in test_dataloader:
                X = batch[0].to(device)
                rating = batch[1].to(device)
                _, p_t = model.get_aspect_importance(X)
                y_pred, _ = model.pyramid(X, p_t)
                preds += y_pred.argmax(dim=1).tolist()
                trues += rating.tolist()

        print("Precision:", precision_score(trues, preds, average='macro'))
        print("Recall:", recall_score(trues, preds, average='macro'))
        print("F1-score:", f1_score(trues, preds, average='macro'))
        print("Accuracy:", accuracy_score(trues, preds))


    def test_acd(self, model, test_set, batch_size, device='cuda'):
        test_dataloader = DataLoader(test_set, batch_size=batch_size, drop_last=True)
        with torch.no_grad():
            for th in np.linspace(0, 1e-2, 20):
                S = 0
                G = 0
                S_G = 0
                for batch in test_dataloader:
                    X = batch[0].to(device)
                    aspects = batch[2].to(device)
                    _, p_t = model.get_aspect_importance(X)
                    
                    for b in range(X.shape[0]):
                        trues, preds = self.test_func.eval_acd(p_t[b].tolist(), aspects[b].tolist(), th)
                        S += len(preds)
                        G += len(trues)
                        S_G += len(list(set(preds).intersection(set(trues))))

                P = S_G / S
                R = S_G / G
                F1 = (2 * P * R) / (P + R)

                print('Th: %0.8f | P: %0.8f | R: %0.8f | F1: %0.8f' % (th, P, R, F1))


                
                
    def test_acsa(self, model, test_set, batch_size, device, best_th):
        test_dataloader = DataLoader(test_set, batch_size=batch_size, drop_last=True)
        acc = 0
        n = 0
        S = 0
        G = 0
        S_G = 0
        TRUE = []
        PRED = []
        with torch.no_grad():
            for batch in test_dataloader:
                X = batch[0].to(device)
                aspects = batch[2].to(device)
                _, p_t = model.get_aspect_importance(X)
                _, a_senti = model.pyramid(X, p_t)
                a_senti = a_senti.argmax(dim=-1)
                
                for b in range(X.shape[0]):
                    trues, preds, preds_c = self.test_func.eval_acsa(p_t[b].tolist(), a_senti[b].tolist(), aspects[b].tolist(), best_th)
                    
                    S += len(preds)
                    G += len(trues)
                    S_G += len(list(set(preds).intersection(set(trues))))
                    
                    # SC
                    for i in range(len(trues)):
                        TRUE.append(trues[i][1])
                        PRED.append(preds_c[i][1])
                        if trues[i] == preds_c[i]:
                            acc += 1
                        n += 1
            
            if S_G != 0:
                P = S_G / S
                R = S_G / G
                F1 = (2 * P * R) / (P + R)
                print('P: %0.5f | R: %0.5f | F1: %0.5f' % (P, R, F1))
            else:
                print(0)
            print("ACSA Accuracy:", acc / n)  
            
            res = pd.DataFrame(confusion_matrix(TRUE, PRED, labels=[0, 1, 2]))
            res.columns = ['p=-1', 'p=0', 'p=1']
            res.index = ['t=-1', 't=0', 't=1']
            print(res)



class End2end_trainer():
    def __init__(self, data_name):
        self.data_name = data_name
        self.lr = 1e-3
        self.step_size = 5

    def train(self, model, train_set, dev_set, device='cuda', epochs=5, batch_size=32, data_name='', model_name=''):
        model.train()
        opt = optim.Adam(model.parameters(), lr=1e-2)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
        Train_Loss_list = []
        Train_Acc_list = []
        Dev_Loss_list = []
        Dev_Acc_list = []
        for e in range(epochs):
            train_generator = generate_batches(train_set, batch_size=batch_size, shuffle=True)
            train_losses = []
            train_acc = []
            epochsize = int(len(train_set) / batch_size)
            with tqdm.trange(epochsize) as pbar:
                for b in pbar: # 每个batch
                    item = next(train_generator)
                    X = item['X'].to(device)
                    asp_senti = item['a'].to(device)

                    y_preds = model(X)
                    l = sum([F.cross_entropy(y_preds[i], asp_senti[:, i]) for i in range(len(y_preds))])
                    opt.zero_grad()
                    l.backward()
                    opt.step()
                    train_losses.append(l.item())

                    lst = [i.argmax(dim=1).tolist() for i in y_preds]
                    y_p = []
                    for b in range(X.shape[0]):
                        y_p.append([i[b] for i in lst])
                    y_p = torch.tensor(y_p).to(device)

                    train_acc.append((y_p == asp_senti).sum().cpu().item()/(X.shape[0]*len(y_preds)))

                    d = 'TRAIN EPOCH: %d | TRAIN-LOSS: %0.5f | TRAIN-ACC: %0.5f' % (e + 1, train_losses[-1], train_acc[-1])
                    pbar.set_description(d)
                    # Saving ABAE model
                    torch.save(model.state_dict(), './model_params/' + str(data_name)  + '_' + str(model_name) + \
                               '_' + str(e+1) +'.model')

            model.eval()
            dev_acc, dev_loss = self.evaluate(model, dev_set, batch_size, device)
            print("VAL-ACC: %.5f" % (dev_acc))
            model.train()

            scheduler.step()

            # plot epoch info
            Train_Loss_list.append(np.mean(train_losses))
            Dev_Loss_list.append(dev_loss)
            Train_Acc_list.append(np.mean(train_acc))
            Dev_Acc_list.append(dev_acc)

        self.plot(Train_Loss_list, Dev_Loss_list, Train_Acc_list, Dev_Acc_list, epochs)
        model.eval()

    # 评估函数
    def evaluate(self, model, data_set, batch_size, device='cuda'):
        data_generator = generate_batches(data_set, batch_size=batch_size, shuffle=True)
        y_t = []
        y_p = []
        dev_losses = []
        epochsize = int(len(data_set) / batch_size)
        with torch.no_grad():
            with tqdm.tqdm(range(epochsize), total=epochsize, desc='validating') as pbar:
                for b in pbar:
                    item = next(data_generator)
                    X = item['X'].to(device)
                    asp_senti = item['a'].to(device)

                    y_preds = model(X)
                    l = sum([F.cross_entropy(y_preds[i], asp_senti[:, i]) for i in range(len(y_preds))])

                    dev_losses.append(l.item())
                    y_t += asp_senti.tolist()
                    lst = [i.argmax(dim=1).tolist() for i in y_preds]
                    y_preds_ = []
                    for b in range(X.shape[0]):
                        y_preds_.append([i[b] for i in lst])

                    y_p += y_preds_

        y_p = torch.tensor(y_p).to(device)
        y_t = torch.tensor(y_t).to(device)

        return (y_p == y_t).sum().cpu().item()/(X.shape[0]*len(y_preds)*epochsize), np.mean(dev_losses)


    def test(self, model, test_set, batch_size, device):
        data_generator = generate_batches(test_set, batch_size=128)
        epochsize = int(len(test_set) / batch_size)
        y_p = []
        y_t = []
        with torch.no_grad():
            with tqdm.tqdm(range(epochsize), total=epochsize, desc='testing') as pbar:
                for b in pbar:
                    item = next(data_generator)
                    X = item['X'].to(device)
                    asp_senti = item['a'].to(device)

                    y_preds = model(X)
                    for b in asp_senti.tolist():
                        y_t += b

                    lst = [i.argmax(dim=1).tolist() for i in y_preds]
                    y_preds = []
                    for b in range(X.shape[0]):
                        y_p += [i[b] for i in lst]
        # ACSA
        S = 0
        G = 0
        S_G = 0
        for i in range(len(y_p)):
            if y_p[i] != 0:
                S += 1
            if y_t[i] != 0:
                G += 1
            if y_p[i] == y_t[i] and y_p[i] != 0:
                S_G += 1

        P = S_G / S
        R = S_G / G
        F1 = (2 * P * R) / (P + R)
        # SC
        n = 0
        acc = 0
        for i in range(len(y_p)):
            if y_t[i] != 0:
                if y_t[i] == y_p[i]:
                    acc += 1
                n += 1

        print('ACSA: P: %0.5f | R: %0.5f | F1: %0.5f' % (P, R, F1))
        print("SC: Accuracy:", acc / n)


    # Plot
    def plot(self, lst1, lst2, lst3, lst4, num_epochs):
        x = [i + 1 for i in range(num_epochs)]
        plt.subplots(figsize=(20, 20))

        plt.subplot(2, 1, 1)
        plt.plot(x, lst1, '.-', label='Train')
        plt.plot(x, lst2, '.-', label='Val')
        plt.xticks([i + 1 for i in range(num_epochs)])
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(x, lst3, '.-', label='Train')
        plt.plot(x, lst4, '.-', label='Val')
        plt.xlabel('epoch')
        plt.ylabel('F1')
        plt.legend()

        plt.show()
        # plt.savefig("model_performance.jpg")

