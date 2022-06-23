import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

PAD = '<pad>'
UNK = '<unk>'
ASPECT = '<aspect>'

PAD_INDEX = 0
UNK_INDEX = 1
ASPECT_INDEX = 2

INF = 1e9

def squash(x, dim=-1):
    squared = torch.sum(x * x, dim=dim, keepdim=True)
    scale = torch.sqrt(squared) / (1.0 + squared)
    return scale * x


def sentence_clip(sentence):
    mask = (sentence != PAD_INDEX)
    sentence_lens = mask.long().sum(dim=1, keepdim=False)
    max_len = sentence_lens.max().item()
    return sentence[:, :max_len]

class CapsuleLoss(nn.Module):

    def __init__(self, smooth=0.1, lamda=0.6):
        super(CapsuleLoss, self).__init__()
        self.smooth = smooth
        self.lamda = lamda

    def forward(self, input, target):
        one_hot = torch.zeros_like(input).to(input.device)
        one_hot = one_hot.scatter(1, target.unsqueeze(-1), 1)
        a = torch.max(torch.zeros_like(input).to(input.device), 1 - self.smooth - input)
        b = torch.max(torch.zeros_like(input).to(input.device), input - self.smooth)
        loss = one_hot * a * a + self.lamda * (1 - one_hot) * b * b
        loss = loss.sum(dim=1, keepdim=False)
        return loss.mean()


class Attention(nn.Module):
    """
    The base class of attention.
    """

    def __init__(self, dropout):
        super(Attention, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        """
        query: FloatTensor (batch_size, query_size) or FloatTensor (batch_size, num_queries, query_size)
        key: FloatTensor (batch_size, time_step, key_size)
        value: FloatTensor (batch_size, time_step, hidden_size)
        mask: ByteTensor (batch_size, time_step) or ByteTensor (batch_size, num_queries, time_step)
        """
        single_query = False
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            single_query = True
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1)
            else:
                assert mask.size(1) == query.size(1)
        score = self._score(query, key) # FloatTensor (batch_size, num_queries, time_step)
        weights = self._weights_normalize(score, mask)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = weights.matmul(value)
        if single_query:
            output = output.squeeze(1)
        return output

    def _score(self, query, key):
        raise NotImplementedError('Attention score method is not implemented.')

    def _weights_normalize(self, score, mask):
        if not mask is None:
            score = score.masked_fill(mask == 0, -INF)
        weights = F.softmax(score, dim=-1)
        return weights

    def get_attention_weights(self, query, key, mask=None):
        single_query = False
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            single_query = True
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1)
            else:
                assert mask.size(1) == query.size(1)
        score = self._score(query, key)  # FloatTensor (batch_size, num_queries, time_step)
        weights = self._weights_normalize(score, mask)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        if single_query:
            weights = weights.squeeze(1)
        return weights


class BilinearAttention(Attention):

    def __init__(self, query_size, key_size, dropout=0):
        super(BilinearAttention, self).__init__(dropout)
        self.weights = nn.Parameter(torch.FloatTensor(query_size, key_size))
        init.xavier_uniform_(self.weights)

    def _score(self, query, key):
        """
        query: FloatTensor (batch_size, num_queries, query_size)
        key: FloatTensor (batch_size, time_step, key_size)
        """
        score = query.matmul(self.weights).matmul(key.transpose(1, 2))
        return score


class CapsuleNetwork(nn.Module):

    def __init__(self, embedding, aspect_embedding, hidden_size, capsule_size, dropout, num_categories):
        super(CapsuleNetwork, self).__init__()
        self.embedding = embedding
        self.aspect_embedding = aspect_embedding
        embed_size = embedding.embedding_dim
        self.capsule_size = capsule_size
        self.aspect_transform = nn.Sequential(
            nn.Linear(embed_size, capsule_size),
            nn.Dropout(dropout)
        )
        self.sentence_transform = nn.Sequential(
            nn.Linear(hidden_size, capsule_size),
            nn.Dropout(dropout)
        )
        self.norm_attention = BilinearAttention(capsule_size, capsule_size)
        self.guide_capsule = nn.Parameter(
            torch.Tensor(num_categories, capsule_size)
        )
        self.guide_weight = nn.Parameter(
            torch.Tensor(capsule_size, capsule_size)
        )
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.capsule_projection = nn.Linear(capsule_size, capsule_size * num_categories)
        self.dropout = dropout
        self.num_categories = num_categories
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.guide_capsule)
        init.xavier_uniform_(self.guide_weight)

    def load_sentiment(self, path):
        sentiment = np.load(path)
        e1 = np.mean(sentiment)
        d1 = np.std(sentiment)
        e2 = 0
        d2 = np.sqrt(2.0 / (sentiment.shape[0] + sentiment.shape[1]))
        sentiment = (sentiment - e1) / d1 * d2 + e2
        self.guide_capsule.data.copy_(torch.tensor(sentiment))

    def forward(self, sentence, aspect):
        # get lengths and masks
        sentence = sentence_clip(sentence)
        sentence_mask = (sentence != PAD_INDEX)
        # embedding
        sentence = self.embedding(sentence)
        sentence = F.dropout(sentence, p=self.dropout, training=self.training)
        aspect = self.embedding(aspect)
        aspect = F.dropout(aspect, p=self.dropout, training=self.training)
        # sentence encode layer
        sentence = self._sentence_encode(sentence, aspect)
        # primary capsule layer
        sentence = self.sentence_transform(sentence)
        primary_capsule = squash(sentence, dim=-1)
        # aspect capsule layer
        aspect = self.aspect_transform(aspect)
        aspect_capsule = squash(aspect, dim=-1)
        # aspect aware normalization
        norm_weight = self.norm_attention.get_attention_weights(aspect_capsule, primary_capsule, sentence_mask)
        # capsule guided routing
        category_capsule = self._capsule_guided_routing(primary_capsule, norm_weight)
        category_capsule_norm = torch.sqrt(torch.sum(category_capsule * category_capsule, dim=-1, keepdim=False))
        return category_capsule_norm

    def _sentence_encode(self, sentence, aspect, mask=None):
        raise NotImplementedError('_sentence_encode method is not implemented.')

    def _capsule_guided_routing(self, primary_capsule, norm_weight):
        guide_capsule = squash(self.guide_capsule)
        guide_matrix = primary_capsule.matmul(self.guide_weight).matmul(guide_capsule.transpose(0, 1))
        guide_matrix = F.softmax(guide_matrix, dim=-1)
        guide_matrix = guide_matrix * norm_weight.unsqueeze(-1) * self.scale  # (batch_size, time_step, num_categories)
        category_capsule = guide_matrix.transpose(1, 2).matmul(primary_capsule)
        category_capsule = F.dropout(category_capsule, p=self.dropout, training=self.training)
        category_capsule = squash(category_capsule)
        return category_capsule


class RecurrentCapsuleNetwork(CapsuleNetwork):
    def __init__(self, embedding, aspect_embedding, num_layers, bidirectional, capsule_size, dropout, num_categories):
        super(RecurrentCapsuleNetwork, self).__init__(
            embedding=embedding,
            aspect_embedding=aspect_embedding,
            hidden_size=embedding.embedding_dim * 2, # bidirectional=True
            capsule_size=capsule_size,
            dropout=dropout,
            num_categories=num_categories
        )
        embed_size = 200
        self.rnn = nn.GRU(
            input_size=embed_size * 2,
            hidden_size=embed_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.bidirectional = bidirectional

    def _sentence_encode(self, sentence, aspect, mask=None):
        batch_size, time_step, embed_size = sentence.size()
        aspect_aware_sentence = torch.cat((
            sentence, aspect.unsqueeze(1).expand(batch_size, time_step, embed_size)
        ), dim=-1)
        output, _ = self.rnn(aspect_aware_sentence)
        if self.bidirectional:
            sentence = sentence.unsqueeze(-1).expand(batch_size, time_step, embed_size, 2)
            sentence = sentence.contiguous().view(batch_size, time_step, embed_size * 2)
        output = output + sentence
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output