import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]


class FocalLoss:
    def __init__(self, alpha_t=None, gamma=0):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        self.alpha_t = torch.tensor(alpha_t) if alpha_t else None
        self.gamma = gamma

    def __call__(self, outputs, targets):
        if self.alpha_t is None and self.gamma == 0:
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets)

        elif self.alpha_t is not None and self.gamma == 0:
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                           weight=self.alpha_t)

        elif self.alpha_t is None and self.gamma != 0:
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()

        elif self.alpha_t is not None and self.gamma != 0:
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                        weight=self.alpha_t, reduction='none')
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss


class attention(nn.Module):
    def __init__(self, d_embed):
        super(attention, self).__init__()
        self.M = nn.Linear(d_embed, d_embed)
        self.M.weight.data.uniform_(-0.1, 0.1)

    def forward(self, e_i):
        y_s = torch.mean(e_i, dim=-1)
        d_i = torch.bmm(e_i.transpose(1, 2), self.M(y_s).unsqueeze(2)).tanh()
        a_i = torch.exp(d_i) / torch.sum(torch.exp(d_i))
        return a_i.squeeze(1)


class ABAE(nn.Module):
    def __init__(self, E, T):
        super(ABAE, self).__init__()
        n_vocab, d_embed = E.shape
        n_aspects, d_embed = T.shape
        self.E = nn.Embedding(n_vocab, d_embed) # E应该是词嵌入矩阵
        self.T = nn.Embedding(n_aspects, d_embed) # 而T应该就是Aspect embedding矩阵
        self.attention = attention(d_embed)
        self.linear = nn.Linear(d_embed, n_aspects)
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False) # 不更新词向量矩阵
        # self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=True)
        self.T.weight = nn.Parameter(torch.from_numpy(T), requires_grad=True)

    def forward(self, pos, negs):
        p_t, z_s = self.get_asp_importances(pos)
        r_s = F.normalize(torch.mm(self.T.weight.t(), p_t.t()).t(), dim=-1) # 重构的表达
        e_n = self.E(negs).transpose(-2, -1)
        z_n = F.normalize(torch.mean(e_n, dim=-1), dim=-1)
        return r_s, z_s, z_n

    def get_asp_importances(self, x):     
        e_i = self.E(x).transpose(1, 2) # 嵌入
        a_i = self.attention(e_i) # 得到注意力
        z_s = F.normalize(torch.bmm(e_i, a_i).squeeze(2), dim=-1) # 得到句子表达
        p_t = F.softmax(self.linear(z_s), dim=1) # p_t就是aspect importance
        return p_t, z_s
    
    def aspects(self):
        E_n = F.normalize(self.E.weight[:-2], dim=1) # 最后两维是<unk>和<pad>
        T_n = F.normalize(self.T.weight, dim=1)
        projection = torch.mm(E_n, T_n.t()).t()
        return projection


class JPAN(nn.Module):
    def __init__(self, E, T, num_hiddens=100, num_layers=3, dropout=0.5):
        super(JPAN, self).__init__()
        n_vocab, d_embed = E.shape
        n_aspects, d_embed = T.shape
        self.E = nn.Embedding(n_vocab, d_embed)  # E应该是词嵌入矩阵
        self.T = nn.Embedding(n_aspects, d_embed)  # 而T应该就是Aspect embedding矩阵
        self.rnn = nn.LSTM(d_embed, num_hiddens, num_layers, bidirectional=True, dropout=dropout)
        self.attention = attention(d_embed)
        self.linear = nn.Linear(d_embed, n_aspects)
        # self.dropout = nn.Dropout(dropout)
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False)  # 不更新词向量矩阵
        self.T.weight = nn.Parameter(torch.from_numpy(T), requires_grad=True)

        # Module: Sentiment Classification
        # self.bilstm = nn.LSTM(input_size=d_embed,
        #                       hidden_size=num_hiddens,
        #                       num_layers=num_layers,
        #                       bidirectional=True,
        #                       dropout=dropout)
        # self.bilstm_decoder = nn.Linear(2*num_hiddens, 3)
        self.decoder = nn.Linear(2 * num_hiddens + d_embed, 3)
        #self.bilstm_decoder = nn.Sequential(
        #    nn.Linear(2 * num_hiddens, 2 * num_hiddens),
        #    nn.ReLU(),
        #    nn.Linear(2 * num_hiddens, 3),
        #)

    def forward(self, pos, negs):
        p_t, z_s = self.get_asp_importances(pos)
        sentiment, aspects_sentiments, _, _ = self.sentiment_predict(pos, p_t)
        r_s = F.normalize(torch.mm(self.T.weight.t(), p_t.t()).t(), dim=-1)  # 重构的表达
        e_n = self.E(negs).transpose(-2, -1)
        z_n = F.normalize(torch.mean(e_n, dim=-1), dim=-1)
        return r_s, z_s, z_n, p_t, sentiment, aspects_sentiments

    def get_asp_importances(self, w2v_x):
        e_i = self.E(w2v_x).transpose(1, 2)  # 嵌入
        a_i = self.attention(e_i)  # 得到注意力
        z_s = F.normalize(torch.bmm(e_i, a_i).squeeze(2), dim=-1)  # 得到句子表达
        p_t = F.softmax(self.linear(z_s), dim=1)  # p_t就是aspect importance
        return p_t, z_s

    def sentiment_predict(self, w2v_x, p_t):
        e = self.E(w2v_x)
        outputs, _ = self.rnn(e.permute(1, 0, 2))
        # w_sentiments = self.bilstm_decoder(outputs).permute(1, 0, 2)  # (batch, seq_len, 3)
        outputs = outputs.permute(1, 0, 2)
        # outputs: [batch_size, seq_len, hidden_size * bidirectional]
        x = torch.cat((outputs, e), 2)
        # x: [batch_size, seq_len, embdding_dim + hidden_size * bidirectional]
        w_sentiments = torch.tanh(self.decoder(x))
        # w_sentiments: [batch_size, seq_len, 3hidden_size]
        '''
        self.T            :      (asp_num * wv_dim)
        p_t               :      (batch_size * asp_num)
        words_sentiment   :      (batch * seq_len * 3)
        '''
        batch_size = w_sentiments.shape[0]
        a_sentiments = []
        sentiments = []
        w_att = []

        for b in range(batch_size):
            '''1.利用asp_emb和word_embedding计算出attention(余弦相似度)'''
            # Cosine
            # words_att = torch.softmax(F.cosine_similarity(e[b].unsqueeze(1), self.T.weight.unsqueeze(0), dim=-1), dim=0)
            # Euclidean distance
            # words_att = torch.softmax(torch.cdist(e[b].unsqueeze(1), self.T.weight.unsqueeze(0)).squeeze(1), dim=0)
            # Dot Product
            words_att = torch.softmax(torch.matmul(e[b], self.T.weight.permute(1, 0)), dim=0)
            w_att.append(words_att)
            # (seq_len * asp_num)

            '''2.结合每个词的sentiment计算出asp的sentiment'''
            words_sentiment_batch = w_sentiments[b]
            asp_sentiment = torch.matmul(words_att.permute(1, 0), words_sentiment_batch)  # (asp_num * 3)
            a_sentiments.append(asp_sentiment)

            '''3.再结合asp_dis得到最终sentiment'''
            sentiment = torch.matmul(p_t[b], asp_sentiment)
            sentiments.append(sentiment)

        a_sentiments = torch.stack(a_sentiments)  # batch_size * asp_num * 3
        w_att = torch.stack(w_att)

        sentiment = torch.stack(sentiments)
        return sentiment, a_sentiments, w_sentiments, w_att

    def aspects(self):
        E_n = F.normalize(self.E.weight[:-2], dim=1)  # 最后两维是<unk>和<pad>
        T_n = F.normalize(self.T.weight, dim=1)
        projection = torch.mm(E_n, T_n.t()).t()
        return projection


class JPAN_BERT(nn.Module):
    def __init__(self, E, T):
        super(JPAN_BERT, self).__init__()
        n_vocab, d_embed = E.shape
        n_aspects, d_embed = T.shape
        self.E = nn.Embedding(n_vocab, d_embed)  # E应该是词嵌入矩阵
        self.T = nn.Embedding(n_aspects, d_embed)  # 而T应该就是Aspect embedding矩阵
        self.attention = attention(d_embed)
        self.linear = nn.Linear(d_embed, n_aspects)
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False)  # 不更新词向量矩阵
        self.T.weight = nn.Parameter(torch.from_numpy(T), requires_grad=True)

        # Module: Sentiment Classification (BERT)
        self.bert = BertModel.from_pretrained("./model_params/new_bert")
        # for param in self.bert.parameters():
        #    param.requires_grad = True

        self.bert_decoder = nn.Linear(768, 3)
        # self.bert_decoder = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.ReLU(),
        #     nn.Linear(768, 3),
        # )

    def forward(self, pos, negs, bert_input):
        p_t, z_s = self.get_asp_importances(pos)
        sentiment, aspects_sentiments = self.sentiment_predict(pos, bert_input, p_t)
        r_s = F.normalize(torch.mm(self.T.weight.t(), p_t.t()).t(), dim=-1)  # 重构的表达
        e_n = self.E(negs).transpose(-2, -1)
        z_n = F.normalize(torch.mean(e_n, dim=-1), dim=-1)
        return r_s, z_s, z_n, p_t, sentiment, aspects_sentiments

    def get_asp_importances(self, w2v_x):
        e_i = self.E(w2v_x).transpose(1, 2)  # 嵌入
        a_i = self.attention(e_i)  # 得到注意力
        z_s = F.normalize(torch.bmm(e_i, a_i).squeeze(2), dim=-1)  # 得到句子表达
        p_t = F.softmax(self.linear(z_s), dim=1)  # p_t就是aspect importance
        return p_t, z_s

    def sentiment_predict(self, w2v_x, bert_input, p_t):
        e = self.E(w2v_x)
        context = bert_input[0]  # 输入的句子
        mask = bert_input[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        hidden, _ = self.bert(context, attention_mask=mask, return_dict=False)
        words_sentiment = self.bert_decoder(hidden)  # [batch_size, seq_len, 3]
        '''
        self.T            :      (asp_num * wv_dim)
        p_t               :      (batch_size * asp_num)      
        words_sentiment   :      (batch * seq_len * 1)
        '''
        batch_size = words_sentiment.shape[0]
        aspects_sentiments = []
        sentiments = []

        for b in range(batch_size):
            '''1.利用asp_emb和word_embedding计算出attention(余弦相似度)'''
            # Cosine
            # words_att = torch.softmax(F.cosine_similarity(e[b].unsqueeze(1), self.T.weight.unsqueeze(0), dim=-1), dim=0)
            # Euclidean distance
            # words_att = torch.softmax(torch.cdist(e[b].unsqueeze(1), self.T.weight.unsqueeze(0)).squeeze(1), dim=0)
            # Dot Product
            words_att = torch.softmax(torch.matmul(e[b], self.T.weight.permute(1, 0)), dim=0)
            # (seq_len * asp_num)

            '''2.结合每个词的sentiment计算出asp的sentiment'''
            words_sentiment_batch = words_sentiment[b][1:-1, :]
            asp_sentiment = torch.matmul(words_att.permute(1, 0), words_sentiment_batch)  # (asp_num * 3)
            aspects_sentiments.append(asp_sentiment)

            '''3.再结合asp_dis得到最终sentiment'''
            sentiment = torch.matmul(p_t[b], asp_sentiment)
            sentiments.append(sentiment)

        aspects_sentiments = torch.stack(aspects_sentiments)  # batch_size * asp_num * 3
        sentiment = torch.stack(sentiments)
        return sentiment, aspects_sentiments

    def aspects(self):
        E_n = F.normalize(self.E.weight[:-2], dim=1)  # 最后两维是<unk>和<pad>
        T_n = F.normalize(self.T.weight, dim=1)
        projection = torch.mm(E_n, T_n.t()).t()
        return projection


class BERT(nn.Module):
    def __init__(self, E, hidden_size=768):
        super(BERT, self).__init__()
        n_vocab, d_embed = E.shape
        self.E = nn.Embedding(n_vocab, d_embed)
        self.bert = BertModel.from_pretrained("./model_params/new_bert")
        #for param in self.bert.parameters():
        #    param.requires_grad = True
        self.fc = nn.Linear(hidden_size, 3)
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, return_dict=False)

        return self.fc(pooled)


# TextRCNN
class TextRCNN(nn.Module):
    def __init__(self, E, hidden_size=100, num_layers=1, dropout=0.5):
        super(TextRCNN, self).__init__()
        n_vocab, d_embed = E.shape
        self.E = nn.Embedding(n_vocab, d_embed)
        self.rnn = nn.LSTM(d_embed, hidden_size, num_layers, bidirectional=True, dropout=dropout)
        self.W2 = Linear(2 * hidden_size + d_embed, hidden_size * 2)
        self.fc = Linear(hidden_size * 2, 3)
        self.dropout = nn.Dropout(dropout)
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False)

    def forward(self, x):
        e = self.E(x)
        outputs, _ = self.rnn(e.permute(1, 0, 2))
        # outputs: [real_seq_len, batch_size, hidden_size * 2]
        outputs = outputs.permute(1, 0, 2)
        # outputs: [batch_size, seq_len, hidden_size * bidirectional]
        x = torch.cat((outputs, e), 2)
        # x: [batch_size, seq_len, embdding_dim + hidden_size * bidirectional]
        y2 = torch.tanh(self.W2(x)).permute(0, 2, 1)
        # y2: [batch_size, hidden_size * bidirectional, seq_len]
        y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2)
        # y3: [batch_size, hidden_size * bidirectional]

        return self.fc(y3)

# TextCNN
class TextCNN(nn.Module):
    def __init__(self, E, n_filters=200, filter_sizes=[1, 2, 3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()
        n_vocab, d_embed = E.shape
        self.E = nn.Embedding(n_vocab, d_embed)
        self.convs = Conv1d(d_embed, n_filters, filter_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 3)
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False)

    def forward(self, x):
        e = self.E(x)
        conved = self.convs(e.permute(0, 2, 1))
        # conv_n = [batch size, n_filters, sent len - filter_sizes[n] - 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)



# BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, E, hidden_size=100, num_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()
        n_vocab, d_embed = E.shape
        self.E = nn.Embedding(n_vocab, d_embed)
        self.rnn = nn.LSTM(input_size=d_embed,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=True,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_size * 4, 3)
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False)

    def forward(self, x):
        e = self.E(x)
        outputs, _ = self.rnn(e.permute(1, 0, 2))
        # outputs: [real_seq_len, batch_size, hidden_size * 2]
        # outputs = outputs.permute(1, 0, 2)
        # outputs: [batch_size, real_seq, hidden_size * 2]
        encoding = torch.cat((outputs[0], outputs[-1]), dim=-1)

        return self.fc(encoding)


# BiLSTM + Attention
class LSTMATT(nn.Module):
    def __init__(self, E, hidden_size=100, num_layers=1, dropout=0.5):
        super(LSTMATT, self).__init__()
        n_vocab, d_embed = E.shape
        self.E = nn.Embedding(n_vocab, d_embed)
        self.rnn = nn.LSTM(input_size=d_embed,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=True,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, 3)

        self.W_w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False)

        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

    def forward(self, x):
        e = self.E(x)
        outputs, _ = self.rnn(e.permute(1, 0, 2))
        # outputs: [real_seq_len, batch_size, hidden_size * 2]
        outputs = outputs.permute(1, 0, 2)
        # outputs: [batch_size, real_seq, hidden_size * 2]
        """ tanh attention 的实现 """
        score = torch.tanh(torch.matmul(outputs, self.W_w))
        # score: [batch_size, real_seq, hidden_size * 2]
        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        # attention_weights: [batch_size, real_seq, 1]
        scored_x = outputs * attention_weights
        # scored_x : [batch_size, real_seq, hidden_size * 2]
        feat = torch.sum(scored_x, dim=1)
        # feat : [batch_size, hidden_size * 2]

        return self.fc(feat)


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