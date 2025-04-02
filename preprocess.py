import torch
import torch.utils.data as Data
import pandas as pd
from transformers import BertModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel
from transformers import AlbertTokenizer, AlbertModel
from utils import set_seed
set_seed(1)


class InitAspectMat():
    def __init__(self, bert_tokenizer, bert_model, data_name):
        self.tokenizer = bert_tokenizer
        self.bert = bert_model
        if data_name == 'Trip':
            self.n_aspects = 7
            # self.seed_words = ['value price quality worth cost expensive $ reasonable pricey cheaper',
            #                    'room suite view bed suite bathroom shower desk well-equipped balcony',
            #                    'location traffic minute restaurant locations mclintock chandler located convenient mall',
            #                    'clean dirty maintain smell spotless tidy roomy neat comfortable decorated',
            #                    'check-in stuff check help reservation check-outs flights appointment doctor tech',
            #                    'service food breakfast buffet staff customer exceptional ambiance friendly experience',
            #                    'business center computer internet businesses biz collier printer desktop wifi']
            self.seed_words = ['value',
                               'room',
                               'location',
                               'clean',
                               'check-in',
                               'service',
                               'business']
        elif data_name == 'ASAP':
            self.n_aspects = 5
            self.seed_words = ['位置',
                               '服务',
                               '价格',
                               '环境',
                               '食物']
            # self.seed_words = ['位置 交通 便利 市中心 靠近 停车场',
            #                    '服务 热情 周到 专业 效率 耐心',
            #                    '价格 合理 昂贵 实惠 性价比 折扣',
            #                    '环境 安静 整洁 舒适 装修 氛围',
            #                    '食物 新鲜 美味 口感 食材 烹饪']
        
        elif data_name == 'MAMS':
            self.n_aspects = 8
            self.seed_words = ['ambience',
                               'food',
                               'menu',
                               'miscellaneous',
                               'place',
                               'price',
                               'service',
                               'staff']
        elif data_name == 'rest_14':
            self.n_aspects = 5
            self.seed_words = {'service',
                               'food',
                               'miscellaneous',
                               'price',
                               'ambience'}
        else: # rest_15, rest_16
            self.n_aspects = 6
            self.seed_words = {'restaurant',
                               'service',
                               'food',
                               'drinks',
                               'ambience',
                               'location'}

    def T(self):
        inputs = torch.tensor([self.tokenizer.encode(seed, add_special_tokens=False, padding="max_length", max_length=5) for seed in self.seed_words])
        T = self.bert(inputs).pooler_output
        
        return T



class TripPreProcess():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('./model_params/bert-base-uncased')
        self.bert = BertModel.from_pretrained("./model_params/bert-base-uncased")
        #self.tokenizer= RobertaTokenizer.from_pretrained("./model_params/roberta-base")
        #self.bert = RobertaModel.from_pretrained("./model_params/roberta-base")
        #self.tokenizer = AlbertTokenizer.from_pretrained('./model_params/albert-base-v2')
        #self.bert = AlbertModel.from_pretrained("./model_params/albert-base-v2")
        self.train_df = pd.read_csv(r"./data/processed/TripDMS_train.csv")
        self.dev_df = pd.read_csv(r"./data/processed/TripDMS_dev.csv")
        self.test_df = pd.read_csv(r"./data/processed/TripDMS_test.csv")
        
        need_col = ['review', 'Overall', 'value', 'room', 'location', 'cleanliness', 'checkin', 'service', 'business']
        self.train_data = self.train_df[need_col].values
        self.dev_data = self.dev_df[need_col].values
        self.test_data = self.test_df[need_col].values
        
        # 初始矩阵T
        init = InitAspectMat(self.tokenizer, self.bert, data_name='Trip')
        self.T = init.T()

    def f(self, data, max_l=100, duplicate=False):
        inputs = torch.tensor([self.tokenizer.encode(text, add_special_tokens=False, max_length=max_l, truncation=True, padding="max_length") for text, _, _, _, _, _, _, _, _ in data])
        ratings = torch.tensor([score for _, score, _, _, _, _, _, _, _ in data])
        aspects = torch.tensor([[V, R, L, C, St, Se, B] for _, _, V, R, L, C, St, Se, B in data])
        return inputs, ratings, aspects

    def get_dataset(self, max_l=100, duplicate=False):
        train_set = Data.TensorDataset(*self.f(self.train_data))
        dev_set = Data.TensorDataset(*self.f(self.dev_data))
        test_set = Data.TensorDataset(*self.f(self.test_data))

        return self.T, train_set, dev_set, test_set
    

class ASAPPreProcess():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('./model_params/bert-base-chinese')
        self.bert = BertModel.from_pretrained("./model_params/bert-base-chinese")
        #self.tokenizer = AutoTokenizer.from_pretrained("./model_params/roberta-base-chinese")
        #self.bert = AutoModel.from_pretrained("./model_params/roberta-base-chinese")
        #self.tokenizer = AutoTokenizer.from_pretrained("./model_params/albert-chinese-base/")
        #self.bert = AlbertModel.from_pretrained("./model_params/albert-chinese-base/")
        
        self.train_df = pd.read_csv(r"./data/processed/ASAP_train.csv")
        self.dev_df = pd.read_csv(r"./data/processed/ASAP_dev.csv")
        self.test_df = pd.read_csv(r"./data/processed/ASAP_test.csv")
        
        need_col = ['review', 'sentiment', 'Location', 'Service', 'Price', 'Ambience', 'Food']
        self.train_data = self.train_df[need_col].values
        self.dev_data = self.dev_df[need_col].values
        self.test_data = self.test_df[need_col].values
        
        # 初始矩阵T
        init = InitAspectMat(self.tokenizer, self.bert, data_name='ASAP')
        self.T = init.T()

    def f(self, data, max_l=100, duplicate=False):
        inputs = torch.tensor([self.tokenizer.encode(text, add_special_tokens=False, max_length=max_l, truncation=True, padding="max_length") for text, _, _, _, _, _, _ in data])
        ratings = torch.tensor([score for _, score, _, _, _, _, _ in data])
        aspects = torch.tensor([[L, S, P, A, F] for _, _, L, S, P, A, F in data])
        return inputs, ratings, aspects

    def get_dataset(self, max_l=100, duplicate=False):
        train_set = Data.TensorDataset(*self.f(self.train_data))
        dev_set = Data.TensorDataset(*self.f(self.dev_data))
        test_set = Data.TensorDataset(*self.f(self.test_data))

        return self.T, train_set, dev_set, test_set
    
    

# class NoRatingPreProcess():
#     def __init__(self, data_name):
#         self.tokenizer = BertTokenizer.from_pretrained('./model_params/bert-base-uncased')
#         self.bert = BertModel.from_pretrained("./model_params/bert-base-uncased")
#         if data_name == 'mams':
#             self.train_df = pd.read_csv(r"./data/processed/xxx.csv")
#             self.dev_df = pd.read_csv(r"./data/processed/xxx.csv")
#             self.test_df = pd.read_csv(r"./data/processed/xxx.csv")
        
#         elif data_name == 'rest_14':
#             self.train_df = pd.read_csv(r"./data/processed/xxx.csv")
#             self.dev_df = pd.read_csv(r"./data/processed/xxx.csv")
#             self.test_df = pd.read_csv(r"./data/processed/xxx.csv")
        
#         else:
#             self.train_df = pd.read_csv(r"./data/processed/xxx.csv")
#             self.dev_df = pd.read_csv(r"./data/processed/xxx.csv")
#             self.test_df = pd.read_csv(r"./data/processed/xxx.csv")
        
#         need_col = ['review', 'Overall', 'value', 'room', 'location', 'cleanliness', 'checkin', 'service', 'business']
#         self.train_data = self.train_df[need_col].values
#         self.dev_data = self.dev_df[need_col].values
#         self.test_data = self.test_df[need_col].values
        
#         # 初始矩阵T
#         init = InitAspectMat(self.tokenizer, self.bert, data_name='Trip')
#         self.T = init.T()

#     def f(self, data, max_l=100, duplicate=False):
#         inputs = torch.tensor([self.tokenizer.encode(text, add_special_tokens=False, max_length=max_l, truncation=True, padding="max_length") for text, _, _, _, _, _, _, _, _ in data])
#         ratings = torch.tensor([score for _, score, _, _, _, _, _, _, _ in data])
#         aspects = torch.tensor([[V, R, L, C, St, Se, B] for _, _, V, R, L, C, St, Se, B in data])
#         return inputs, ratings, aspects

#     def get_dataset(self, max_l=200, duplicate=False):
#         train_set = Data.TensorDataset(*self.f(self.train_data))
#         dev_set = Data.TensorDataset(*self.f(self.dev_data))
#         test_set = Data.TensorDataset(*self.f(self.test_data))

#         return self.T, train_set, dev_set, test_set