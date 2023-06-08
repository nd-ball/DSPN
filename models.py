import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x


class ABAE(nn.Module):
    def __init__(self, T, bert_model, bert_tokenizer, d_embed=768):
        super(ABAE, self).__init__()        
        self.bert = bert_model
        self.tokenizer = bert_tokenizer
        
        n_aspects = T.shape[0]
        self.T = nn.Embedding(n_aspects, d_embed)
        self.linear = nn.Linear(d_embed, n_aspects)
        self.T.weight = nn.Parameter(T, requires_grad=False)
        

    def forward(self, pos, negs):
        z_s, p_t = self.get_aspect_importance(pos)
        r_s = F.normalize(torch.mm(self.T.weight.t(), p_t.t()).t(), dim=-1) # 重构的表达
        z_n = torch.stack([self.bert(b).pooler_output for b in negs])
        
        return r_s, z_s, z_n, p_t
    
    
    def get_aspect_importance(self, pos):
        z_s = self.bert(pos).pooler_output # 句子表达
        p_t = F.softmax(self.linear(z_s), dim=1) # 方面权重
        
        return z_s, p_t


class BERT(nn.Module):
    def __init__(self, bert_model, d_embed=768):
        super(BERT, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(d_embed, 3)

    def forward(self, x):
        return self.fc(self.bert(x).pooler_output)



class DSPN(nn.Module):
    def __init__(self, T, bert_model, bert_tokenizer, d_embed=768):
        super(DSPN, self).__init__()
        self.bert = bert_model
        self.tokenizer = bert_tokenizer
        n_aspects = T.shape[0]
        self.T = nn.Embedding(n_aspects, d_embed)
        self.linear = nn.Linear(d_embed, n_aspects)
        self.T.weight = nn.Parameter(T, requires_grad=False)
        self.decoder = nn.Linear(768, 3)
        # self.decoder = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.ReLU(),
        #     nn.Linear(768, 3),
        # )

    def forward(self, pos, negs):
        z_s, p_t = self.get_aspect_importance(pos)
        r_senti, a_senti = self.pyramid(pos, p_t)
        r_s = F.normalize(torch.mm(self.T.weight.t(), p_t.t()).t(), dim=-1) # 重构的表达
        z_n = torch.stack([self.bert(b).pooler_output for b in negs])
        
        return r_s, z_s, z_n, p_t, r_senti, a_senti

    
    
    def get_aspect_importance(self, pos):
        z_s = self.bert(pos).pooler_output # 句子表达
        p_t = F.softmax(self.linear(z_s), dim=1) # 方面权重
        
        return z_s, p_t
    
    

    def pyramid(self, pos, p_t):
        e_n = self.bert(pos).last_hidden_state # batch × seq_len × 768
        w_senti = torch.tanh(self.decoder(e_n)) # batch × seq_len × 3
        '''
        self.T            :      (n_aspects × d_embed)
        p_t               :      (batch × n_aspects)
        words_sentiment   :      (batch × seq_len * 3)
        '''
        a_senti = [] # aspect-level sentiment prediction
        r_senti = [] # review-level sentiment prediction
        w_att = []

        for b in range(w_senti.shape[0]):
            '''1.利用aspect embedding和word embedding计算出对每个aspect的word-level attention'''
            # Cosine
            # words_att = torch.softmax(F.cosine_similarity(e[b].unsqueeze(1), self.T.weight.unsqueeze(0), dim=-1), dim=0)
            # Euclidean distance
            # words_att = torch.softmax(torch.cdist(e[b].unsqueeze(1), self.T.weight.unsqueeze(0)).squeeze(1), dim=0)
            # Dot Product
            words_att = torch.softmax(torch.matmul(e_n[b], self.T.weight.permute(1, 0)), dim=0) # (seq_len × asp_num)
            w_att.append(words_att)
            

            '''2.结合每个词的sentiment计算出asp的sentiment'''
            words_sentiment_batch = w_senti[b]
            asp_sentiment = torch.matmul(words_att.permute(1, 0), words_sentiment_batch)  # (asp_num * 3)
            a_senti.append(asp_sentiment)

            '''3.再结合asp_dis得到最终sentiment'''
            sentiment = torch.matmul(p_t[b], asp_sentiment)
            r_senti.append(sentiment)

        a_senti = torch.stack(a_senti)  # batch_size * asp_num * 3
        w_att = torch.stack(w_att)

        r_senti = torch.stack(r_senti)
        return r_senti, a_senti






class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, E, T):
        super(CNN_Gate_Aspect_Text, self).__init__()
        V, D = E.shape
        C = 3
        A = T.shape[0]
        Co = 100 # kernel_num
        Ks = (3,4,5) # kernel_size

        self.embed = nn.Embedding(V, D)
        self.embed.weight = nn.Parameter(torch.from_numpy(E), requires_grad=True)
        self.aspect_embed = nn.Embedding(A, D)
        self.aspect_embed.weight = nn.Parameter(torch.from_numpy(T), requires_grad=True)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])

        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc_aspect = nn.Linear(D, Co)

    def forward(self, feature, order):
        feature = self.embed(feature)  # (N, L, D)
        aspect_v = self.aspect_embed.weight[order]
        # aspect_v = self.aspect_embed(aspect)  # (N, L', D)
        # aspect_v = aspect_v.sum(1) / aspect_v.size(1)

        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)]

        # pooling method
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]

        x0 = torch.cat(x0, 1)
        logit = self.fc1(x0)  # (N,C)
        return logit


# end2end LSTM
class E2E_LSTM(nn.Module):
    def __init__(self, E, data_name, hidden_size=50, bidirectional=True, dropout=0.5):
        super(E2E_LSTM, self).__init__()
        n_vocab, d_embed = E.shape
        self.data_name = data_name
        self.E = nn.Embedding(n_vocab, d_embed)
        self.encoder = nn.LSTM(input_size=d_embed,
                           hidden_size=hidden_size,
                           num_layers=1,
                           bidirectional=True,
                           dropout=dropout)
        
        self.fc1 = nn.Linear(4*hidden_size, 4)
        self.fc2 = nn.Linear(4*hidden_size, 4)
        self.fc3 = nn.Linear(4*hidden_size, 4)
        self.fc4 = nn.Linear(4*hidden_size, 4)
        self.fc5 = nn.Linear(4*hidden_size, 4)
        self.fc6 = nn.Linear(4*hidden_size, 4)
        self.fc7 = nn.Linear(4*hidden_size, 4)
        self.dropout = nn.Dropout(dropout)
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False) 

    def forward(self, x):
        e = self.E(x) # embedding
        e = self.dropout(e) # dropout
        outputs, _ = self.encoder(e.permute(1, 0, 2))
        v = torch.cat((outputs[0], outputs[-1]), dim=-1) # batch_size, 4 * hidden_size
        if self.data_name == 'ASAP':
            return self.fc1(v), self.fc2(v), self.fc3(v), self.fc4(v), self.fc5(v)
        elif self.data_name == 'TA':
            return self.fc1(v), self.fc2(v), self.fc3(v), self.fc4(v), self.fc5(v), self.fc6(v), self.fc7(v)
        elif self.data_name =='GS':
            return self.fc1(v), self.fc2(v), self.fc3(v), self.fc4(v), self.fc5(v)
        


# end2end CNN
class E2E_CNN(nn.Module):
    def __init__(self, E, data_name, n_filters=300, filter_sizes=[3,4,5], dropout=0.5):
        super(E2E_CNN, self).__init__()
        n_vocab, d_embed = E.shape
        self.E = nn.Embedding(n_vocab, d_embed)
        self.data_name = data_name
        self.encoder = Conv1d(d_embed, n_filters, filter_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes) * n_filters, 4)
        self.fc2 = nn.Linear(len(filter_sizes) * n_filters, 4)
        self.fc3 = nn.Linear(len(filter_sizes) * n_filters, 4)
        self.fc4 = nn.Linear(len(filter_sizes) * n_filters, 4)
        self.fc5 = nn.Linear(len(filter_sizes) * n_filters, 4)
        self.fc6 = nn.Linear(len(filter_sizes) * n_filters, 4)
        self.fc7 = nn.Linear(len(filter_sizes) * n_filters, 4)
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False)
        
    def forward(self, x):
        e = self.E(x)
        conved = self.encoder(e.permute(0, 2, 1))
        # conv_n = [batch size, n_filters, sent len - filter_sizes[n] - 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        v = self.dropout(torch.cat(pooled, dim=1))
        # v = [batch size, n_filters * len(filter_sizes)]
        if self.data_name == 'ASAP':
            return self.fc1(v), self.fc2(v), self.fc3(v), self.fc4(v), self.fc5(v)
        elif self.data_name == 'TA':
            return self.fc1(v), self.fc2(v), self.fc3(v), self.fc4(v), self.fc5(v), self.fc6(v), self.fc7(v)
        elif self.data_name =='GS':
            return self.fc1(v), self.fc2(v), self.fc3(v), self.fc4(v), self.fc5(v)