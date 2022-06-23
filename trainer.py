import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from pytorch_pretrained_bert import BertAdam
from preprocess.CoherenceScore import coherence_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from test_func import test_func
import matplotlib.pyplot as plt


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


def sample_aspects(projection, i2w, n=8):
        projection = torch.sort(projection, dim=1)
        for j, (projs, index) in enumerate(zip(*projection)):
            index = index[-n:].detach().cpu().numpy()
            words = ', '.join([i2w[i] for i in index])
            print('Aspect %2d: %s' % (j, words))


def generate_batches(dataset, batch_size, shuffle=False, drop_last=True, bert=False):
    """A generator function which wraps the PyTorch DataLoader. """
    dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    if bert:
        out_dict = {}
        for datalist in dataloader:
            out_dict['X_w2v'] = datalist[0]
            out_dict['X_bert'] = datalist[1]
            out_dict['y'] = datalist[2]
            out_dict['a'] = datalist[3]
            out_dict['seq_len'] = datalist[4]
            out_dict['mask'] = datalist[5]
            yield out_dict
    else:
        out_dict = {}
        for datalist in dataloader:
            out_dict['X'] = datalist[0]
            out_dict['y'] = datalist[1]
            out_dict['a'] = datalist[2]
            yield out_dict


class ABAE_trainer():
    def __init__(self, data_name):
        print("Preparing...")
        if data_name == 'ASAP':
            self.co_score = coherence_score("./data/processed/ASAP_comments.txt")
            self.test_fx = test_func(data_name=data_name)
        elif data_name == 'TA':
            self.co_score = coherence_score("./data/processed/TripAdvisor_comments.txt")
            self.test_fx = test_func(data_name=data_name)
        elif data_name == 'GS':
            self.co_score = coherence_score("./data/processed/GreatSchools_comments.txt")
            self.test_fx = test_func(data_name=data_name)
        else:
            print("Not Supported")

    def evaluate_aspects(self, projection, i2w, data_name, model_name, epoch_num, n_aspects, n=50):
        co_scores = []
        projection = torch.sort(projection, dim=1)
        with open("./record/" + str(data_name) + "_" + str(model_name) + "_aspects_record.txt", 'a+', encoding='utf-8') as f:
            f.write("===============Epoch" + str(epoch_num) + "=================\n")
            for j, (projs, index) in enumerate(zip(*projection)):
                f.write("Aspect " + str(j + 1) + ":\n")
                index = index[-n:].detach().cpu().numpy()
                words = [i2w[i] for i in index]
                for w in words[:20]:
                    f.write(str(w) + " ")
                f.write("\n")
                co_scores.append([self.co_score.get_co_score(words, i) for i in [10, 20, 30, 40, 50]])
                # print('Aspect %2d: %s' % (j, ', '.join(words)))
        co_scores = torch.Tensor(co_scores)
        return (torch.sum(co_scores, dim=0) / n_aspects).tolist()

    def train(self, model, w2v_model, train_set, dev_set, device='cuda', epochs=5, batch_size=32, negsize=20,
              ortho_reg=0.1, data_name='', model_name=''):
        i2w = dict((w2v_model.w2i[w], w) for w in w2v_model.w2i)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
        model.train()
        Train_Loss_list = []
        Val_Loss_list = []
        for e in range(epochs):
            train_generator = generate_batches(train_set, batch_size=batch_size, shuffle=True)
            neg_generator = generate_batches(train_set, batch_size=batch_size, shuffle=False)
            train_losses = []
            epochsize = int(len(train_set) / batch_size)
            with tqdm.trange(epochsize) as pbar:
                for b in pbar: # 每个batch
                    X = next(train_generator)['X']
                    X_neg = next(neg_generator)['X']
                    neg_samples = torch.stack(tuple([X_neg[torch.randperm(X_neg.shape[0])[:negsize]] for _ in range(batch_size)]))
                    
                    X = X.to(device)
                    neg_samples = neg_samples.to(device)
                    
                    r_s, z_s, z_n = model(X, neg_samples)
                    J = max_margin_loss(r_s, z_s, z_n)
                    U = orthogonal_regularization(model.T.weight)
                    loss = J + ortho_reg * batch_size * U                
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
                    train_losses.append(loss.item())
                    x = (e + 1, train_losses[-1])
                    d = 'TRAIN EPOCH: %d | TRAIN-LOSS: %0.5f' % x
                    pbar.set_description(d)
                    # Saving ABAE model
                    torch.save(model.state_dict(), './model_params/' + str(data_name)  + '_' + str(model_name) + \
                               '_' + str(e+1) +'.model')
            
            model.eval()
            val_loss = self.evaluate(model, dev_set, device, batch_size, negsize, ortho_reg)
            model.train()

            scheduler.step()

            Train_Loss_list.append(np.mean(train_losses))
            Val_Loss_list.append(val_loss)
            co_scores = self.evaluate_aspects(model.aspects(), i2w, data_name, model_name, epoch_num=e+1,
                                         n_aspects=w2v_model.n_aspects)
            print("Coherence Score(10->50):", co_scores)
            # sample_aspects(model.aspects(), i2w)


        self.plot(Train_Loss_list, Val_Loss_list, epochs)
        model.eval()

    def evaluate(self, model, dev_set, device='cuda', batch_size=32, negsize=20, ortho_reg=0.1):
        dev_generator = generate_batches(dev_set, batch_size=batch_size, shuffle=True)
        neg_generator = generate_batches(dev_set, batch_size=batch_size, shuffle=False)
        
        losses = []
        epochsize = int(len(dev_set) / batch_size)
        with torch.no_grad():
            with tqdm.tqdm(range(epochsize), total=epochsize, desc='validating') as pbar:
                for b in pbar:
                    X = next(dev_generator)['X']
                    X_neg = next(neg_generator)['X']
                    neg_samples = torch.stack(tuple([X_neg[torch.randperm(X_neg.shape[0])[:negsize]] for _ in range(batch_size)]))

                    X = X.to(device)
                    neg_samples = neg_samples.to(device)
                    r_s, z_s, z_n = model(X, neg_samples)
                    J = max_margin_loss(r_s, z_s, z_n).item()
                    U = orthogonal_regularization(model.T.weight).item()
                    losses.append((J + ortho_reg * batch_size * U))
                    x = (b + 1, np.mean(losses))
                    pbar.set_description('VAL BATCH: %d | VAL-LOSS: %0.5f' % x)
        return np.mean(losses)

    def test(self, model, test_set, batch_size, device='cuda'):
        epochsize = int(len(test_set) / batch_size)
        with torch.no_grad():
            for th in np.linspace(0.01, 0.1):
                S = 0
                G = 0
                S_G = 0
                test_generator = generate_batches(test_set, batch_size=batch_size, shuffle=False)
                for i in range(epochsize):
                    item = next(test_generator)
                    X = item['X']
                    asp_info = item['a']
                    X = X.to(device)
                    p_t, _ = model.get_asp_importances(X)
                    for b in range(X.shape[0]):
                        g_ai, l_ai = self.test_fx.evaluate_asp_identification(p_t[b].tolist(), asp_info[b].tolist(), th)
                        S += len(l_ai)
                        G += len(g_ai)
                        S_G += len(list(set(g_ai).intersection(set(l_ai))))

                P = S_G / S
                R = S_G / G
                F1 = (2 * P * R) / (P + R)

                print('Th: %0.5f | P: %0.5f | R: %0.5f | F1: %0.5f' % (th, P, R, F1))

    def plot(self, lst1, lst2, num_epochs):
        x = [i + 1 for i in range(num_epochs)]

        plt.plot(x, lst1, '.-', label='Train')
        plt.plot(x, lst2, '.-', label='Val')
        plt.xticks([i + 1 for i in range(num_epochs)])
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        # plt.savefig("model_performance.jpg")



class SC_trainer():
    def __init__(self, data_name):
        self.data_name = data_name
        self.loss = nn.CrossEntropyLoss()
        self.lr = 1e-3
        self.step_size = 5
        

    def train(self, model, train_set, dev_set, device='cuda', epochs=5, batch_size=32, data_name='', model_name=''):
        model.train()
        opt = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=self.step_size, gamma=0.1)
        Train_Loss_list = []
        Val_Loss_list = []
        Train_f1_list = []
        Dev_f1_list = []
        for e in range(epochs):
            train_generator = generate_batches(train_set, batch_size=batch_size, shuffle=True)
            train_losses = []
            train_f1 = []
            epochsize = int(len(train_set) / batch_size)
            with tqdm.trange(epochsize) as pbar:
                for b in pbar:  # 每个batch
                    item = next(train_generator)
                    X = item['X'].to(device)
                    y = item['y'].to(device)
                    y_pred = model(X)
                    l = self.loss(y_pred, y)
                    opt.zero_grad()
                    l.backward()
                    opt.step()
                    train_losses.append(l.item())
                    y_t = y.tolist()
                    y_p = y_pred.argmax(dim=1).tolist()
                    # acc = (y_pred.argmax(dim=1) == y).sum().cpu().item()/y.shape[0]
                    train_f1.append(f1_score(y_t, y_p, average='macro'))
                    d = 'TRAIN EPOCH: %d | TRAIN-LOSS: %0.5f | TRAIN-F1: %0.5f' % (
                    e + 1, train_losses[-1], train_f1[-1])
                    pbar.set_description(d)
                    # Saving ABAE model
                    torch.save(model.state_dict(), './model_params/' + str(data_name) + '_' + str(model_name) + \
                               '_' + str(e + 1) + '.model')

            model.eval()
            dev_f1, dev_loss = self.evaluate(model, dev_set, batch_size, device)
            print("VAL-F1: %.5f" % (dev_f1))
            model.train()
            
            scheduler.step()

            # plot epoch info
            Train_Loss_list.append(np.mean(train_losses))
            Val_Loss_list.append(dev_loss)
            Train_f1_list.append(np.mean(train_f1))
            Dev_f1_list.append(dev_f1)

        self.plot(Train_Loss_list, Val_Loss_list, Train_f1_list, Dev_f1_list, epochs)
        model.eval()

    # 评估函数
    def evaluate(self, model, data_set, batch_size, device='cuda'):
        data_generator = generate_batches(data_set, batch_size=batch_size, shuffle=True)
        y_t = []
        y_p = []
        losses = []
        epochsize = int(len(data_set) / batch_size)
        with torch.no_grad():
            with tqdm.tqdm(range(epochsize), total=epochsize, desc='validating') as pbar:
                for b in pbar:
                    item = next(data_generator)
                    X = item['X'].to(device)
                    y = item['y'].to(device)
                    y_pred = model(X)
                    losses.append(self.loss(y_pred, y).item())
                    y_t += y.tolist()
                    y_p += y_pred.argmax(dim=1).tolist()

        return f1_score(y_t, y_p, average='macro'), np.mean(losses)

    def test(self, model, test_set, batch_size, device):
        data_generator = generate_batches(test_set, batch_size)
        epochsize = int(len(test_set) / batch_size)
        y_preds = []
        y_trues = []
        with torch.no_grad():
            with tqdm.tqdm(range(epochsize), total=epochsize, desc='testing') as pbar:
                for b in pbar:
                    item = next(data_generator)
                    X = item['X'].to(device)
                    y = item['y'].to(device)
                    y_pred = model(X)

                    y_preds += y_pred.argmax(dim=1).tolist()
                    y_trues += y.tolist()

        print("Precision:", precision_score(y_trues, y_preds, average='macro'))
        print("Recall:", recall_score(y_trues, y_preds, average='macro'))
        print("F1-score:", f1_score(y_trues, y_preds, average='macro'))
        print("Accuracy:", accuracy_score(y_trues, y_preds))


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



class SC_BERT_trainer():
    def __init__(self, data_name):
        self.data_name = data_name
        self.step_size = 5

    def train(self, model, train_set, dev_set, device, batch_size, epochs=20, data_name='', model_name=''):
        model.train()
        opt1 = optim.Adam(model.parameters(), lr=0.0001)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        opt2 = BertAdam(optimizer_grouped_parameters,
                             lr=5e-5,
                             warmup=0.05,
                             t_total=len(train_set) * epochs)

        scheduler = optim.lr_scheduler.StepLR(opt2, step_size=self.step_size, gamma=0.1)


        Train_Loss_list = []
        Train_f1_list = []
        Dev_Loss_list = []
        Dev_f1_list = []
        for epoch in range(epochs):
            train_generator = generate_batches(train_set, batch_size=batch_size, shuffle=True, bert=True)
            epochsize = int(len(train_set) / batch_size)
            with tqdm.trange(epochsize) as pbar:
                for b in pbar:  # 每个batch
                    item = next(train_generator)
                    X = item['X_bert'].to(device)
                    labels = item['y'].to(device)
                    seq_len = item['seq_len'].to(device)
                    mask = item['mask'].to(device)

                    outputs = model((X, seq_len, mask))
                    model.zero_grad()
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    opt2.step()

            model.eval()
            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_f1 = f1_score(true, predic, average='macro')
            # train_acc = accuracy_score(true, predic)
            dev_f1, dev_loss = self.evaluate(model, dev_set, batch_size, device)
            model.train()

            scheduler.step()

            Train_Loss_list.append(loss.item())
            Dev_Loss_list.append(dev_loss)
            Train_f1_list.append(train_f1)
            Dev_f1_list.append(dev_f1)

            torch.save(model.state_dict(),
                       './model_params/' + str(data_name) + '_'+ str(model_name) +'_' + str(epoch + 1) + '.model')
            print('EPOCH: %d | TRAIN LOSS: %0.5f | TRAIN-F1: %0.5f | VAL LOSS: %0.5f | VAL-F1: %0.5f'
                  % (epoch + 1, loss.item(), train_f1, dev_loss, dev_f1))
        self.plot(Train_Loss_list, Dev_Loss_list, Train_f1_list, Dev_f1_list, epochs)
        model.train()

    def evaluate(self, model, data_set, batch_size, device):
        data_generator = generate_batches(data_set, batch_size=batch_size, shuffle=True, bert=True)
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        epochsize = int(len(data_set) / batch_size)
        with torch.no_grad():
            with tqdm.tqdm(range(epochsize), total=epochsize, desc='validating') as pbar:
                for b in pbar:
                    item = next(data_generator)
                    X = item['X_bert'].to(device)
                    labels = item['y'].to(device)
                    seq_len = item['seq_len'].to(device)
                    mask = item['mask'].to(device)

                    outputs = model((X, seq_len, mask))
                    loss = F.cross_entropy(outputs, labels)
                    loss_total += loss
                    labels = labels.data.cpu().numpy()
                    predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                    labels_all = np.append(labels_all, labels)
                    predict_all = np.append(predict_all, predic)

        # acc = accuracy_score(labels_all, predict_all)
        f1 = f1_score(labels_all, predict_all, average='macro')

        return f1, loss_total / len(data_set)

    def test(self, model, test_set, batch_size, device):
        data_generator = generate_batches(test_set, batch_size, bert=True)
        epochsize = int(len(test_set) / batch_size)
        y_preds = []
        y_trues = []
        with torch.no_grad():
            with tqdm.tqdm(range(epochsize), total=epochsize, desc='testing') as pbar:
                for b in pbar:
                    item = next(data_generator)
                    X = item['X_bert'].to(device)
                    y = item['y'].to(device)
                    seq_len = item['seq_len'].to(device)
                    mask = item['mask'].to(device)

                    y_pred = model((X, seq_len, mask))
                    y_preds += y_pred.argmax(dim=1).tolist()
                    y_trues += y.tolist()

        print("Precision:", precision_score(y_trues, y_preds, average='macro'))
        print("Recall:", recall_score(y_trues, y_preds, average='macro'))
        print("F1-score:", f1_score(y_trues, y_preds, average='macro'))
        print("Accuracy:", accuracy_score(y_trues, y_preds))


    # Plot
    def plot(self, lst1, lst2, lst3, lst4, num_epochs):
        x = [i + 1 for i in range(num_epochs)]
        plt.subplots(figsize=(20, 20))

        plt.subplot(2, 1, 1)
        plt.plot(x, lst1, '.-', label='Train')
        plt.plot(x, lst2, '.-', label='Val')
        plt.xticks([i + 1 for i in range(num_epochs)])
        plt.xlabel('epoch')
        plt.ylabel('Loss of ABAE')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(x, lst3, '.-', label='Train')
        plt.plot(x, lst4, '.-', label='Val')
        plt.xlabel('epoch')
        plt.ylabel('Loss of Prediction')
        plt.legend()

        plt.show()
        # plt.savefig("model_performance.jpg")



class JPAN_trainer():
    def __init__(self, data_name):
        print("Preparing...")
        self.data_name = data_name
        self.loss_senti = nn.CrossEntropyLoss()
        self.lr = 1e-3
        self.step_size = 5
        if self.data_name == 'ASAP':
            self.co_score = coherence_score("./data/processed/ASAP_comments.txt")
            self.test_fx = test_func(data_name=data_name)
        elif self.data_name == 'TA':
            self.co_score = coherence_score("./data/processed/TripAdvisor_comments.txt")
            self.test_fx = test_func(data_name=data_name)
        elif data_name == 'GS':
            self.co_score = coherence_score("./data/processed/GreatSchools_comments.txt")
            self.test_fx = test_func(data_name=data_name)
        else:
            print("Not Supported")

    def train(self, model, w2v_model, train_set, dev_set, device='cuda', epochs=5, batch_size=32, negsize=20, ortho_reg=0.1,
              data_name='', model_name=''):
        i2w = dict((w2v_model.w2i[w], w) for w in w2v_model.w2i)
        opt = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=self.step_size, gamma=0.1)

        Train_Loss_a_list = []
        Train_Loss_s_list = []
        Train_f1_list = []
        Dev_Loss_a_list = []
        Dev_Loss_s_list = []
        Dev_f1_list = []
        model.train()
        for e in range(epochs):
            train_generator = generate_batches(train_set, batch_size=batch_size, shuffle=True)
            neg_generator = generate_batches(train_set, batch_size=batch_size, shuffle=False)
            train_a_losses = []
            train_s_losses = []
            train_losses = []
            train_f1 = []
            epochsize = int(len(train_set) / batch_size)
            with tqdm.trange(epochsize) as pbar:
                for b in pbar:  # 每个batch
                    item = next(train_generator)
                    w2v_X = item['X'].to(device)
                    y = item['y'].to(device)
                    X_neg = next(neg_generator)['X']
                    neg_samples = torch.stack(
                        tuple([X_neg[torch.randperm(X_neg.shape[0])[:negsize]] for _ in range(batch_size)]))
                    neg_samples = neg_samples.to(device)

                    r_s, z_s, z_n, p_t, y_pred, asp_senti_pred = model(w2v_X, neg_samples)
                    J = max_margin_loss(r_s, z_s, z_n)
                    U = orthogonal_regularization(model.T.weight)

                    loss_ab = J + ortho_reg * batch_size * U
                    loss_s = self.loss_senti(y_pred, y)

                    l = 0.01 * loss_ab + loss_s
                    opt.zero_grad()
                    l.backward()
                    opt.step()

                    train_a_losses.append(loss_ab.item())
                    train_s_losses.append(loss_s.item())
                    train_losses.append(l.item())
                    train_f1.append(f1_score(y.tolist(), y_pred.argmax(dim=1).tolist(), average='macro'))
                    # acc = (y_pred.argmax(dim=1) == y).sum().cpu().item() / y.shape[0]
                    # train_acc.append(acc)
                    x = (e + 1, train_losses[-1], train_a_losses[-1], train_s_losses[-1], train_f1[-1])
                    d = 'EPOCH: %d | LOSS INFO: TOTAL:%0.5f, AB: %0.5f, SC: %0.5f | TRAIN-F1: %0.5f' % x
                    pbar.set_description(d)
                    # Saving ABAE model
                    torch.save(model.state_dict(), './model_params/' + str(data_name) + '_' + str(model_name) + \
                               '_' + str(e + 1) + '.model')

            model.eval()
            dev_f1, dev_loss_ab, dev_loss_s = self.evaluate(model, train_set, dev_set, batch_size, negsize, device)
            print("VAL-F1: %.5f | VAL-LOSS-AB: %.5f | VAL-LOSS-S: %.5f" % (dev_f1, dev_loss_ab, dev_loss_s))
            model.train()
            
            scheduler.step()

            co_scores = self.evaluate_aspects(model.aspects(), i2w, data_name, model_name, epoch_num=e + 1,
                                         n_aspects=w2v_model.n_aspects)
            # sample_aspects(model.aspects(), i2w)
            print("Coherence Score(10->50):", co_scores)
            Train_Loss_a_list.append(np.mean(train_a_losses))
            Train_Loss_s_list.append(np.mean(train_s_losses))
            Train_f1_list.append(np.mean(train_f1))
            Dev_Loss_a_list.append(dev_loss_ab)
            Dev_Loss_s_list.append(dev_loss_s)
            Dev_f1_list.append(dev_f1)

        self.plot(Train_Loss_a_list, Dev_Loss_a_list, Train_Loss_s_list, Dev_Loss_s_list, Train_f1_list, Dev_f1_list, epochs)
        model.eval()

    # 评估函数
    def evaluate(self, model, train_set, dev_set, batch_size, negsize, device='cuda'):
        data_generator = generate_batches(dev_set, batch_size=batch_size, shuffle=True)
        neg_generator = generate_batches(train_set, batch_size=batch_size, shuffle=False)
        y_t = []
        y_p = []
        loss_a_list = []
        loss_s_list = []
        epochsize = int(len(dev_set) / batch_size)
        with torch.no_grad():
            with tqdm.tqdm(range(epochsize), total=epochsize, desc='validating') as pbar:
                for b in pbar:
                    item = next(data_generator)
                    w2v_X = item['X'].to(device)
                    X_neg = next(neg_generator)['X']
                    neg_samples = torch.stack(
                        tuple([X_neg[torch.randperm(X_neg.shape[0])[:negsize]] for _ in range(batch_size)]))
                    neg_samples = neg_samples.to(device)
                    y = item['y'].to(device)

                    r_s, z_s, z_n, p_t, y_pred, asp_senti_pred = model(w2v_X, neg_samples)
                    J = max_margin_loss(r_s, z_s, z_n)
                    U = orthogonal_regularization(model.T.weight)
                    loss_ab = J + 0.1 * batch_size * U  # + ortho_reg * U_pt
                    loss_s = self.loss_senti(y_pred, y)
                    loss_a_list.append(loss_ab.item())
                    loss_s_list.append(loss_s.item())

                    y_t += y.tolist()
                    y_p += y_pred.argmax(dim=1).tolist()

        return f1_score(y_t, y_p, average='macro'), np.mean(loss_a_list), np.mean(loss_s_list)

    # review-level sentiment classification
    def test_review_level_SC(self, model, test_set, batch_size, device):
        test_generator = generate_batches(test_set, batch_size=batch_size, shuffle=False)
        epochsize = int(len(test_set) / batch_size)
        y_trues = []
        y_preds = []
        with torch.no_grad():
            for i in range(epochsize):
                item = next(test_generator)
                X = item['X'].to(device)
                y = item['y'].to(device)
                p_t, _ = model.get_asp_importances(X)
                y_pred, _, _, _ = model.sentiment_predict(X, p_t)
                y_preds += y_pred.argmax(dim=1).tolist()
                y_trues += y.tolist()
        
        print("Precision:", precision_score(y_trues, y_preds, average='macro'))
        print("Recall:", recall_score(y_trues, y_preds, average='macro'))
        print("F1-score:", f1_score(y_trues, y_preds, average='macro'))
        print("Accuracy:", accuracy_score(y_trues, y_preds))

    # aspect idenfitication
    def test_ACD(self, model, test_set, batch_size, device):
        epochsize = int(len(test_set) / batch_size)
        with torch.no_grad():
            for th in np.linspace(0.01, 0.1):
                S = 0
                G = 0
                S_G = 0
                for i in range(epochsize):
                    test_generator = generate_batches(test_set, batch_size=batch_size, shuffle=False)
                    item = next(test_generator)
                    X = item['X'].to(device)
                    asp_info = item['a'].to(device)
                    p_t, _ = model.get_asp_importances(X)
                    _, asp_senti, _, _ = model.sentiment_predict(X, p_t)
                    for b in range(X.shape[0]):
                        g_ai, l_ai = self.test_fx.evaluate_asp_identification(p_t[b].tolist(), asp_info[b].tolist(), th)
                        S += len(l_ai)
                        G += len(g_ai)
                        S_G += len(list(set(g_ai).intersection(set(l_ai))))
                P = S_G / S
                R = S_G / G
                F1 = (2 * P * R) / (P + R)
                # F1 = f1_score(y_ai_trues, y_ai_preds, average='weighted')
                # Accuracy = accuracy_score(y_ai_trues, y_ai_preds)
                print('Th: %0.5f | P: %0.5f | R: %0.5f | F1: %0.9f' % (th, P, R, F1))


    # ACSA
    def test_ACSA(self, model, test_set, batch_size, device, best_th):
        epochsize = int(len(test_set) / batch_size)
        acc = 0
        n = 0
        S = 0
        G = 0
        S_G = 0
        pred = []
        true = []
        test_generator = generate_batches(test_set, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for i in range(epochsize):
                item = next(test_generator)
                X = item['X'].to(device)
                asp_info = item['a'].to(device)
                p_t, _ = model.get_asp_importances(X)
                y_pred, asp_senti, _, _ = model.sentiment_predict(X, p_t)

                for b in range(X.shape[0]):
                    g_as, l_as_ac, l_as_sc = self.test_fx.evaluate_ACSA(p_t[b].tolist(), asp_senti[b].tolist(), asp_info[b].tolist(), best_th)
                    # ACSA
                    S += len(l_as_ac)
                    G += len(g_as)
                    S_G += len(list(set(g_as).intersection(set(l_as_ac))))
                    # SC
                    for i in range(len(g_as)):
                        print(g_as[i])
                        print(l_as_ac[i])
                        print(l_as_sc[i])
                        true.append(g_as[i][1])
                        pred.append(l_as_sc[i][1])
                        if g_as[i] == l_as_sc[i]:
                            acc += 1
                        n += 1
        P = S_G / S
        R = S_G / G
        F1 = (2 * P * R) / (P + R)
        print('ACSA: P: %0.5f | R: %0.5f | F1: %0.5f' % (P, R, F1))
        print("SC: Accuracy:", acc / n)
            

        res = pd.DataFrame(confusion_matrix(true, pred, labels=[-2, -1, 0, 1]))
        res.columns = ['p=-2', 'p=-1', 'p=0', 'p=1']
        res.index = ['t=-2', 't=-1', 't=0', 't=1']
        print(res)


    def evaluate_aspects(self, projection, i2w, data_name, model_name, epoch_num, n_aspects, n=50):
        co_scores = []
        projection = torch.sort(projection, dim=1)
        with open("./record/" + str(data_name) + "_" + str(model_name) + "_aspects_record.txt", 'a+', encoding='utf-8') as f:
            f.write("===============Epoch" + str(epoch_num) + "=================\n")
            for j, (projs, index) in enumerate(zip(*projection)):
                f.write("Aspect " + str(j + 1) + ":\n")
                index = index[-n:].detach().cpu().numpy()
                words = [i2w[i] for i in index]
                for w in words[:20]:
                    f.write(str(w) + " ")
                f.write("\n")
                co_scores.append([self.co_score.get_co_score(words, i) for i in [10, 20, 30, 40, 50]])
                # print('Aspect %2d: %s' % (j, ', '.join(words)))
        co_scores = torch.Tensor(co_scores)
        return (torch.sum(co_scores, dim=0) / n_aspects).tolist()

    # Plot
    def plot(self, lst1, lst2, lst3, lst4, lst5, lst6, num_epochs):
        x = [i + 1 for i in range(num_epochs)]
        plt.subplots(figsize=(20, 20))

        plt.subplot(3, 1, 1)
        plt.plot(x, lst1, '.-', label='Train')
        plt.plot(x, lst2, '.-', label='Val')
        plt.xticks([i + 1 for i in range(num_epochs)])
        plt.xlabel('epoch')
        plt.ylabel('Loss of ABAE')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(x, lst3, '.-', label='Train')
        plt.plot(x, lst4, '.-', label='Val')
        plt.xlabel('epoch')
        plt.ylabel('Loss of Prediction')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(x, lst5, '.-', label='Train')
        plt.plot(x, lst6, '.-', label='Val')
        plt.xlabel('epoch')
        plt.ylabel('F1')
        plt.legend()

        plt.show()
        # plt.savefig("model_performance.jpg")


    # for visualization
    def output_attention(self, model, test_set, device, best_th):
        batch_size = len(test_set)
        test_generator = generate_batches(test_set, batch_size=batch_size, shuffle=False)
        item = next(test_generator)
        X = item['X'].to(device)
        asp_info = item['a'].to(device)
        y = item['y'].to(device)

        flag1 = []
        flag2 = []
        ac_gold = []
        ac_pred = []
        with torch.no_grad():
            p_t, _ = model.get_asp_importances(X)
            r_senti, asp_senti, w_senti, word_att = model.sentiment_predict(X, p_t)
            for b in range(X.shape[0]):
                g_as, _, l_as_sc = self.test_fx.evaluate_ACSA(p_t[b].tolist(), asp_senti[b].tolist(), asp_info[b].tolist(), best_th)
                ac_gold.append(g_as)
                ac_pred.append(l_as_sc)
                # print(g_as[:3])
                # print(l_as_sc[:3])
                # print("============")
                if g_as == l_as_sc:
                    flag1.append(1)
                else:
                    flag1.append(0)

                if r_senti.argmax(dim=1)[b].tolist() == y[b].tolist():
                    flag2.append(1)
                else:
                    flag2.append(0)

            print(len(flag1))
            print(len(flag2))
            print(len([i for i in flag1 if i == 1]))
            print(len([i for i in flag2 if i == 1]))
            
            return y, r_senti, ac_gold, ac_pred, w_senti, word_att, p_t, flag1, flag2



class JPAN_BERT_trainer():
    def __init__(self, data_name='TA'):
        print("Preparing...")
        self.data_name = data_name
        self.loss_senti = nn.CrossEntropyLoss()
        # self.co_score = coherence_score("./data/processed/TripAdvisor_comments.txt")
        self.test_fx = test_func(data_name=data_name)
        self.step_size = 5

    def train(self, model, w2v_model, train_set, dev_set, device='cuda', epochs=5, batch_size=32, negsize=20,
              ortho_reg=0.1, data_name='', model_name=''):
        i2w = dict((w2v_model.w2i[w], w) for w in w2v_model.w2i)
        opt1 = optim.Adam(model.parameters(), lr=0.0001)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        opt2 = BertAdam(optimizer_grouped_parameters,
                        lr=5e-5,
                        warmup=0.05,
                        t_total=len(train_set) * epochs)
        scheduler = optim.lr_scheduler.StepLR(opt2, step_size=self.step_size, gamma=0.1)
        Train_Loss_a_list = []
        Train_Loss_s_list = []
        Train_f1_list = []
        Dev_Loss_a_list = []
        Dev_Loss_s_list = []
        Dev_f1_list = []
        model.train()
        for e in range(epochs):
            train_generator = generate_batches(train_set, batch_size=batch_size, shuffle=True, bert=True)
            neg_generator = generate_batches(train_set, batch_size=batch_size, shuffle=False, bert=True)
            train_a_losses = []
            train_s_losses = []
            train_losses = []
            train_f1 = []
            epochsize = int(len(train_set) / batch_size)
            with tqdm.trange(epochsize) as pbar:
                for b in pbar:  # 每个batch
                    item = next(train_generator)
                    w2v_X = item['X_w2v'].to(device)
                    bert_X = item['X_bert'].to(device)
                    seq_len = item['seq_len'].to(device)
                    mask = item['mask'].to(device)

                    y = item['y'].to(device)
                    X_neg = next(neg_generator)['X_w2v']
                    neg_samples = torch.stack(
                        tuple([X_neg[torch.randperm(X_neg.shape[0])[:negsize]] for _ in range(batch_size)]))
                    neg_samples = neg_samples.to(device)

                    r_s, z_s, z_n, p_t, y_pred, asp_senti_pred = model(w2v_X, neg_samples, (bert_X, seq_len, mask))
                    J = max_margin_loss(r_s, z_s, z_n)
                    U = orthogonal_regularization(model.T.weight)

                    loss_ab = J + ortho_reg * batch_size * U
                    loss_s = self.loss_senti(y_pred, y)

                    l = 0.01 * loss_ab + loss_s
                    opt2.zero_grad()
                    l.backward()
                    opt2.step()

                    train_a_losses.append(loss_ab.item())
                    train_s_losses.append(loss_s.item())
                    train_losses.append(l.item())
                    train_f1.append(f1_score(y.tolist(), y_pred.argmax(dim=1).tolist(), average='macro'))
                    # acc = (y_pred.argmax(dim=1) == y).sum().cpu().item() / y.shape[0]
                    # train_acc.append(acc)
                    x = (e + 1, train_losses[-1], train_a_losses[-1], train_s_losses[-1], train_f1[-1])
                    d = 'EPOCH: %d | LOSS INFO: TOTAL:%0.5f, AB: %0.5f, SC: %0.5f | TRAIN-F1: %0.5f' % x
                    pbar.set_description(d)
                    # Saving ABAE model
                    torch.save(model.state_dict(), './model_params/' + str(data_name) + '_' + str(model_name) + \
                               '_' + str(e + 1) + '.model')

            model.eval()
            dev_f1, dev_loss_ab, dev_loss_s = self.evaluate(model, dev_set, batch_size, device)
            print("VAL-F1: %.5f | VAL-LOSS-AB: %.5f | VAL-LOSS-S: %.5f" % (dev_f1, dev_loss_ab, dev_loss_s))
            model.train()

            scheduler.step()
            co_scores = self.evaluate_aspects(model.aspects(), i2w, data_name, model_name, epoch_num=e + 1,
                                         n_aspects=w2v_model.n_aspects)
            # sample_aspects(model.aspects(), i2w)
            print("Coherence Score(10->50):", co_scores)

            # plot epoch info
            Train_Loss_a_list.append(np.mean(train_a_losses))
            Train_Loss_s_list.append(np.mean(train_s_losses))
            Train_f1_list.append(np.mean(train_f1))
            Dev_Loss_a_list.append(dev_loss_ab)
            Dev_Loss_s_list.append(dev_loss_s)
            Dev_f1_list.append(dev_f1)


        self.plot(Train_Loss_a_list, Dev_Loss_a_list, Train_Loss_s_list, Dev_Loss_s_list, Train_f1_list, Dev_f1_list, epochs)
        model.eval()

    # 评估函数
    def evaluate(self, model, data_set, batch_size, device='cuda'):
        data_generator = generate_batches(data_set, batch_size=batch_size, shuffle=True, bert=True)
        neg_generator = generate_batches(data_set, batch_size=batch_size, shuffle=False, bert=True)
        y_t = []
        y_p = []
        loss_a_list = []
        loss_s_list = []
        epochsize = int(len(data_set) / batch_size)
        with torch.no_grad():
            with tqdm.tqdm(range(epochsize), total=epochsize, desc='validating') as pbar:
                for b in pbar:
                    item = next(data_generator)
                    w2v_X = item['X_w2v'].to(device)
                    bert_X = item['X_bert'].to(device)
                    seq_len = item['seq_len'].to(device)
                    mask = item['mask'].to(device)
                    y = item['y'].to(device)
                    X_neg = next(neg_generator)['X_w2v']
                    neg_samples = torch.stack(
                        tuple([X_neg[torch.randperm(X_neg.shape[0])[:20]] for _ in range(batch_size)]))
                    neg_samples = neg_samples.to(device)

                    r_s, z_s, z_n, p_t, y_pred, asp_senti_pred = model(w2v_X, neg_samples, (bert_X, seq_len, mask))
                    J = max_margin_loss(r_s, z_s, z_n)
                    U = orthogonal_regularization(model.T.weight)

                    ##
                    loss_ab = J + 0.1 * batch_size * U  # + ortho_reg * U_pt
                    loss_s = self.loss_senti(y_pred, y)
                    loss_a_list.append(loss_ab.item())
                    loss_s_list.append(loss_s.item())

                    y_t += y.tolist()
                    y_p += y_pred.argmax(dim=1).tolist()

        return f1_score(y_t, y_p, average='macro'), np.mean(loss_a_list), np.mean(loss_s_list)

    def evaluate_aspects(self, projection, i2w, data_name, model_name, epoch_num, n_aspects, n=50):
        co_scores = []
        projection = torch.sort(projection, dim=1)
        with open("./" + str(data_name) + "_" + str(model_name) + "_aspects_record.txt", 'a+', encoding='utf-8') as f:
            f.write("===============Epoch" + str(epoch_num) + "=================\n")
            for j, (projs, index) in enumerate(zip(*projection)):
                f.write("Aspect " + str(j + 1) + ":\n")
                index = index[-n:].detach().cpu().numpy()
                words = [i2w[i] for i in index]
                for w in words[:20]:
                    f.write(str(w) + " ")
                f.write("\n")
                co_scores.append([self.co_score.get_co_score(words, i) for i in [10, 20, 30, 40, 50]])
                # print('Aspect %2d: %s' % (j, ', '.join(words)))
        co_scores = torch.Tensor(co_scores)
        return (torch.sum(co_scores, dim=0) / n_aspects).tolist()

    # review-level sentiment classification
    def test_review_level_SC(self, model, test_set, batch_size, device):
        test_generator = generate_batches(test_set, batch_size=batch_size, shuffle=False, bert=True)
        epochsize = int(len(test_set) / batch_size)
        y_trues = []
        y_preds = []
        with torch.no_grad():
            for i in range(epochsize):
                item = next(test_generator)
                X_w2v = item['X_w2v'].to(device)
                X_bert = item['X_bert'].to(device)
                seq_len = item['seq_len'].to(device)
                mask = item['mask'].to(device)

                y = item['y'].to(device)
                p_t, _ = model.get_asp_importances(X_w2v)
                y_pred, _ = model.sentiment_predict(X_w2v, (X_bert, seq_len, mask), p_t)
                y_preds += y_pred.argmax(dim=1).tolist()
                y_trues += y.tolist()
        
        print("Precision:", precision_score(y_trues, y_preds, average='macro'))
        print("Recall:", recall_score(y_trues, y_preds, average='macro'))
        print("F1-score:", f1_score(y_trues, y_preds, average='macro'))
        print("Accuracy:", accuracy_score(y_trues, y_preds))

    # aspect idenfitication
    def test_ACD(self, model, test_set, batch_size, device):
        with torch.no_grad():
            for th in np.linspace(0.01, 0.1):
                S = 0
                G = 0
                S_G = 0
                test_generator = generate_batches(test_set, batch_size=batch_size, shuffle=False, bert=True)
                epochsize = int(len(test_set) / batch_size)
                for i in range(epochsize):
                    item = next(test_generator)
                    X_w2v = item['X_w2v'].to(device)
                    X_bert = item['X_bert'].to(device)
                    seq_len = item['seq_len'].to(device)
                    mask = item['mask'].to(device)
                    asp_info = item['a'].to(device)

                    p_t, _ = model.get_asp_importances(X_w2v)
                    _, asp_senti = model.sentiment_predict(X_w2v, (X_bert, seq_len, mask), p_t)

                    for b in range(X_w2v.shape[0]):
                        g_ai, l_ai = self.test_fx.evaluate_asp_identification(p_t[b].tolist(), asp_info[b].tolist(), th)
                        S += len(l_ai)
                        G += len(g_ai)
                        S_G += len(list(set(g_ai).intersection(set(l_ai))))

                P = S_G / S
                R = S_G / G
                F1 = (2 * P * R) / (P + R)
                # F1 = f1_score(y_ai_trues, y_ai_preds, average='weighted')
                # Accuracy = accuracy_score(y_ai_trues, y_ai_preds)
                print('Th: %0.5f | P: %0.5f | R: %0.5f | F1: %0.9f' % (th, P, R, F1))

    # ACSA
    def test_ACSA(self, model, test_set, batch_size, device, best_th):
        epochsize = int(len(test_set) / batch_size)
        acc = 0
        n = 0
        S = 0
        G = 0
        S_G = 0
        pred = []
        true = []
        with torch.no_grad():
            test_generator = generate_batches(test_set, batch_size=batch_size, shuffle=False, bert=True)
            for i in range(epochsize):
                item = next(test_generator)
                X_w2v = item['X_w2v'].to(device)
                X_bert = item['X_bert'].to(device)
                seq_len = item['seq_len'].to(device)
                mask = item['mask'].to(device)
                asp_info = item['a'].to(device)
                p_t, _ = model.get_asp_importances(X_w2v)
                y_pred, asp_senti = model.sentiment_predict(X_w2v, (X_bert, seq_len, mask), p_t)

                for b in range(X_w2v.shape[0]):
                    g_as, l_as_ac, l_as_sc = self.test_fx.evaluate_ACSA(p_t[b].tolist(), asp_senti[b].tolist(),
                                                                        asp_info[b].tolist(), best_th)
                    # ACSA
                    S += len(l_as_ac)
                    G += len(g_as)
                    S_G += len(list(set(g_as).intersection(set(l_as_ac))))
                    # SC
                    for i in range(len(g_as)):
                        true.append(g_as[i][1])
                        pred.append(l_as_sc[i][1])
                        if g_as[i] == l_as_sc[i]:
                            acc += 1
                        n += 1
        P = S_G / S
        R = S_G / G
        F1 = (2 * P * R) / (P + R)
        print('ACSA: P: %0.5f | R: %0.5f | F1: %0.5f' % (P, R, F1))
        print("SC: Accuracy:", acc / n)

        res = pd.DataFrame(confusion_matrix(true, pred, labels=[-2, -1, 0, 1]))
        res.columns = ['p=-2', 'p=-1', 'p=0', 'p=1']
        res.index = ['t=-2', 't=-1', 't=0', 't=1']
        print(res)

    # Plot
    def plot(self, lst1, lst2, lst3, lst4, lst5, lst6, num_epochs):
        x = [i + 1 for i in range(num_epochs)]
        plt.subplots(figsize=(20, 20))

        plt.subplot(3, 1, 1)
        plt.plot(x, lst1, '.-', label='Train')
        plt.plot(x, lst2, '.-', label='Val')
        plt.xticks([i + 1 for i in range(num_epochs)])
        plt.xlabel('epoch')
        plt.ylabel('Loss of ABAE')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(x, lst3, '.-', label='Train')
        plt.plot(x, lst4, '.-', label='Val')
        plt.xlabel('epoch')
        plt.ylabel('Loss of Prediction')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(x, lst5, '.-', label='Train')
        plt.plot(x, lst6, '.-', label='Val')
        plt.xlabel('epoch')
        plt.ylabel('F1')
        plt.legend()

        plt.show()
        # plt.savefig("model_performance.jpg")


class ACSA_supervised_model_trainer():
    def __init__(self, data_name='ASAP'):
        self.loss = nn.CrossEntropyLoss()
        self.test_fx = test_func(data_name=data_name)
        self.lr = 1e-2
        self.step_size = 5
        if data_name == 'ASAP':
            self.lr = 1e-2
            self.step_size = 5
        elif data_name == 'TA':
            self.lr = 1e-3
            self.step_size = 3
        elif data_name == 'GS':
            self.lr = 1e-3
            self.step_size = 5

    def train(self, model, train_set, dev_set, device='cuda', epochs=5, batch_size=32, data_name='', model_name=''):
        model.train()
        opt = optim.Adagrad(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=self.step_size, gamma=0.1)
        Train_Loss_list = []
        Val_Loss_list = []
        Train_acc_list = []
        Dev_acc_list = []
        for e in range(epochs):
            train_generator = generate_batches(train_set, batch_size=batch_size, shuffle=True)
            train_losses = []
            train_acc = []
            epochsize = int(len(train_set) / batch_size)
            with tqdm.trange(epochsize) as pbar:
                for b in pbar:  # 每个batch
                    item = next(train_generator)
                    X = item['X'].to(device)
                    o = item['y'].to(device)
                    asp_senti = item['a'].to(device)

                    y_pred = model(X, o)
                    l = self.loss(y_pred, asp_senti)
                    opt.zero_grad()
                    l.backward()
                    opt.step()
                    train_losses.append(l.item())

                    # train_f1.append(f1_score(y_t, y_p, average='macro'))
                    train_acc.append((y_pred.argmax(dim=1) == asp_senti).sum().cpu().item() / asp_senti.shape[0])
                    d = 'TRAIN EPOCH: %d | TRAIN-LOSS: %0.5f | TRAIN-ACC: %0.5f' % (
                    e + 1, train_losses[-1], train_acc[-1])
                    pbar.set_description(d)
                    # Saving ABAE model
                    torch.save(model.state_dict(), './model_params/' + str(data_name) + '_' + str(model_name) + \
                               '_' + str(e + 1) + '.model')

            model.eval()
            dev_acc, dev_loss = self.evaluate(model, dev_set, batch_size, device)
            print("VAL-ACC: %.5f" % (dev_acc))
            model.train()

            scheduler.step()

            # plot epoch info
            Train_Loss_list.append(np.mean(train_losses))
            Val_Loss_list.append(dev_loss)
            Train_acc_list.append(np.mean(train_acc))
            Dev_acc_list.append(dev_acc)

        self.plot(Train_Loss_list, Val_Loss_list, Train_acc_list, Dev_acc_list, epochs)
        model.eval()

    # 评估函数
    def evaluate(self, model, data_set, batch_size, device='cuda'):
        data_generator = generate_batches(data_set, batch_size=batch_size, shuffle=True)
        y_t = []
        y_p = []
        losses = []
        epochsize = int(len(data_set) / batch_size)
        with torch.no_grad():
            with tqdm.tqdm(range(epochsize), total=epochsize, desc='validating') as pbar:
                for b in pbar:
                    item = next(data_generator)
                    X = item['X'].to(device)
                    o = item['y'].to(device)
                    asp_senti = item['a'].to(device)
                    y_pred = model(X, o)
                    l = self.loss(y_pred, asp_senti)
                    losses.append(l.item())
                    y_t += asp_senti.tolist()
                    y_p += y_pred.argmax(dim=1).tolist()

        return accuracy_score(y_t, y_p), np.mean(losses)


    def test(self, model, test_set, batch_size, device):
        data_generator = generate_batches(test_set, batch_size=batch_size)
        epochsize = int(len(test_set) / batch_size)
        y_preds = []
        y_trues = []
        with torch.no_grad():
            with tqdm.tqdm(range(epochsize), total=epochsize, desc='testing') as pbar:
                for b in pbar:
                    item = next(data_generator)
                    X = item['X'].to(device)
                    o = item['y'].to(device)
                    asp_senti = item['a'].to(device)
                    y_pred = model(X, o)

                    y_preds += y_pred.argmax(dim=1).tolist()
                    y_trues += asp_senti.tolist()
        # SC
        n = 0
        acc = 0
        for i in range(len(y_preds)):
            if y_trues[i] != 0:
                if y_trues[i] == y_preds[i]:
                    acc += 1
                n += 1
        print("SC: Accuracy:", acc / n)


    # Plot
    def plot(self, lst1, lst2, lst3, lst4, num_epochs):
        x = [i + 1 for i in range(num_epochs)]
        plt.subplots(figsize=(20, 20))

        plt.subplot(2, 1, 1)
        plt.plot(x, lst1, '.-', label='Train')
        plt.plot(x, lst2, '.-', label='Test')
        plt.xticks([i + 1 for i in range(num_epochs)])
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(x, lst3, '.-', label='Train')
        plt.plot(x, lst4, '.-', label='Test')
        plt.xlabel('epoch')
        plt.ylabel('Train ACC')
        plt.legend()

        plt.show()
        # plt.savefig("model_performance.jpg")



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
