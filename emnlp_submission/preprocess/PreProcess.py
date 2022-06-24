import torch
import torch.utils.data as Data
import pandas as pd
from preprocess.Word2vec import word2vec
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from transformers import BertModel, BertTokenizer

def tokenizer(text):
    return [tok.replace('\n', '').lower() for tok in text.split(' ')]


class ASAP_PreProcess():
    def __init__(self, sample=False, over_up=0, duplicate=False, end2end=False):
        if duplicate:
            df = pd.read_csv(r"./data/processed/New2_ASAP.csv")
            train_df = df[df['split'] == 'train']
            dev_df = df[df['split'] == 'dev']
            test_df = df[df['split'] == 'test']
            if sample:
                train_df = train_df.head(500)
                dev_df = dev_df.head(500)
                test_df = test_df.head(500)
            if over_up == 1:
                re = RandomOverSampler(random_state=0)
                y = train_df.pop('sentiment')
                X = train_df
                train_df, y_resampled = re.fit_resample(X, y)
                train_df['sentiment'] = y_resampled
            elif over_up == 2:
                re = RandomUnderSampler(random_state=0)
                y = train_df.pop('sentiment')
                X = train_df
                train_df, y_resampled = re.fit_resample(X, y)
                train_df['sentiment'] = y_resampled
            need_col = ['process_review', 'asp', 'asp_senti']
            self.train_data = train_df[need_col].values
            self.dev_data = dev_df[need_col].values
            self.test_data = test_df[need_col].values
        else:
            train_df = pd.read_csv(r"./data/processed/ASAP_train.csv")
            dev_df = pd.read_csv(r"./data/processed/ASAP_dev.csv")
            test_df = pd.read_csv(r"./data/processed/ASAP_test.csv")
            if sample:
                train_df = train_df.head(500)
                dev_df = dev_df.head(500)
                test_df = test_df.head(500)
            if end2end:
                col = ['Location', 'Service', 'Price', 'Ambience', 'Food']
                train_df[col] = train_df[col] + 2
                dev_df[col] = dev_df[col] + 2
                test_df[col] = test_df[col] + 2
            if over_up == 1:
                re = RandomOverSampler(random_state=0)
                y = train_df.pop('sentiment')
                X = train_df
                train_df, y_resampled = re.fit_resample(X, y)
                train_df['sentiment'] = y_resampled
            elif over_up == 2:
                re = RandomUnderSampler(random_state=0)
                y = train_df.pop('sentiment')
                X = train_df
                train_df, y_resampled = re.fit_resample(X, y)
                train_df['sentiment'] = y_resampled
            need_col = ['process_review', 'sentiment', 'Location', 'Service', 'Price', 'Ambience', 'Food']
            self.train_data = train_df[need_col].values
            self.dev_data = dev_df[need_col].values
            self.test_data = test_df[need_col].values

        sentences = train_df['process_review'].values
        w2v_path = './model_params/ASAP_comments_18asp.txt.w2v'
        self.w2v_model = word2vec(sentences)
        self.w2v_model.embed(w2v_path, d_embed=200, min_count=10)
        self.w2v_model.aspect(n_aspects=18)
        x = (self.w2v_model.n_vocab, self.w2v_model.d_embed, self.w2v_model.n_aspects)
        print('N_vocab: %d | D_embed: %d | N_aspects: %d' % x)

    def get_tokenized(self, data, duplicate=False):
        if duplicate:
            return [tokenizer(review) for review, _, _ in data]
        else:
            return [tokenizer(review) for review, _, _, _, _, _, _ in data]

    def preprocess(self, data, max_l, duplicate=False):
        def pad(x):
            return x[:max_l] if len(x) > max_l else x + [self.w2v_model.w2i['<pad>']] * (max_l - len(x))
        f = lambda w: (self.w2v_model.w2i[w] if w in self.w2v_model.w2i else self.w2v_model.w2i['<unk>'])
        if duplicate:
            tokenized_data = self.get_tokenized(data, duplicate)
            features = torch.tensor([pad([f(word) for word in words]) for words in tokenized_data])
            orders = torch.tensor([o for _, o, _ in data])
            asp_sentis = torch.tensor([s for _, _, s in data])
            return features, orders, asp_sentis
        else:
            tokenized_data = self.get_tokenized(data)
            features = torch.tensor([pad([f(word) for word in words]) for words in tokenized_data])
            labels = torch.tensor([score for _, score, _, _, _, _, _ in data])
            asp_sentis = torch.tensor([[L, S, P, A, F] for _, _, L, S, P, A, F in data])

            return features, labels, asp_sentis

    def get_dataset(self, max_l=200, duplicate=False):
        train_set = Data.TensorDataset(*self.preprocess(self.train_data, max_l, duplicate))
        dev_set = Data.TensorDataset(*self.preprocess(self.dev_data, max_l, duplicate))
        test_set = Data.TensorDataset(*self.preprocess(self.test_data, max_l, duplicate))

        return self.w2v_model, train_set, dev_set, test_set


class TA_PreProcess():
    def __init__(self, sample=False, bert=False, duplicate=False, end2end=False):
        if duplicate:
            df = pd.read_csv(r"./data/processed/TripDMS_for_super.csv")
            train_df = df[df['split'] == 'train']
            dev_df = df[df['split'] == 'dev']
            test_df = df[df['split'] == 'test']
            if sample:
                train_df = train_df.head(500)
                dev_df = dev_df.head(500)
                test_df = test_df.head(500)
            need_col = ['process_review', 'asp', 'asp_senti']
            self.train_data = train_df[need_col].values
            self.dev_data = dev_df[need_col].values
            self.test_data = test_df[need_col].values
        else:
            train_df = pd.read_excel(r"./data/processed/TripDMS_train.xlsx")
            dev_df = pd.read_excel(r"./data/processed/TripDMS_dev.xlsx")
            test_df = pd.read_excel(r"./data/processed/TripDMS_test.xlsx")
            if sample:
                train_df = train_df.head(500)
                dev_df = dev_df.head(500)
                test_df = test_df.head(500)
            if end2end:
                col = ['value', 'room', 'location', 'clean', 'stuff', 'service', 'business']
                train_df[col] = train_df[col] + 2
                dev_df[col] = dev_df[col] + 2
                test_df[col] = test_df[col] + 2

            need_col = ['process_review', 'Overall', 'value', 'room', 'location', 'clean', 'stuff', 'service', 'business']
            self.train_data = train_df[need_col].values
            self.dev_data = dev_df[need_col].values
            self.test_data = test_df[need_col].values

        sentences = train_df['process_review'].values
        w2v_path = './model_params/TA_comments_20asp.txt.w2v'
        self.w2v_model = word2vec(sentences)
        self.w2v_model.embed(w2v_path, d_embed=200, min_count=5)
        self.w2v_model.aspect(n_aspects=20)
        x = (self.w2v_model.n_vocab, self.w2v_model.d_embed, self.w2v_model.n_aspects)
        print('N_vocab: %d | D_embed: %d | N_aspects: %d' % x)

        if bert:
            self.tokenizer = BertTokenizer.from_pretrained('./model_params/bert/')
            self.bert_model = BertModel.from_pretrained("./model_params/bert")

            new_toks = []
            for t in self.w2v_model.i2w.values():
                if self.tokenizer.convert_tokens_to_ids(t) == self.tokenizer.unk_token_id:
                    new_toks.append(t)


            num_added_toks = self.tokenizer.add_tokens(new_toks)
            print('We have added', num_added_toks, 'tokens')
            self.bert_model.resize_token_embeddings(len(self.tokenizer))
            # save
            self.tokenizer.save_pretrained("./model_params/new_bert/")
            self.bert_model.save_pretrained("./model_params/new_bert/")


    def get_tokenized(self, data, duplicate=False):
        # data: list of ['processed_text', 'level', 'asp_info']
        if duplicate:
            return [tokenizer(review) for review, _, _ in data]
        else:
            return [tokenizer(review) for review, _, _, _, _, _, _, _, _ in data]

    def preprocess(self, data, max_l=200, duplicate=False):
        def pad(x):
            return x[:max_l] if len(x) > max_l else x + [self.w2v_model.w2i['<pad>']] * (max_l - len(x))
        f = lambda w: (self.w2v_model.w2i[w] if w in self.w2v_model.w2i else self.w2v_model.w2i['<unk>'])
        if duplicate:
            tokenized_data = self.get_tokenized(data, duplicate)
            features = torch.tensor([pad([f(word) for word in words]) for words in tokenized_data])
            orders = torch.tensor([o for _, o, _ in data])
            asp_sentis = torch.tensor([s for _, _, s in data])
            return features, orders, asp_sentis
        else:
            tokenized_data = self.get_tokenized(data)
            features = torch.tensor([pad([f(word) for word in words]) for words in tokenized_data])
            labels = torch.tensor([score for _, score, _, _, _, _, _, _, _ in data])
            asp_sentis = torch.tensor([[V, R, L, C, St, Se, B] for _, _, V, R, L, C, St, Se, B in data])
            return features, labels, asp_sentis

    def preprocess_bert(self, data, max_l=200):
        w2v_ids = []
        bert_ids = []
        labels = []
        asp_info = []
        seq_lens = []
        masks = []
        for i in range(data.shape[0]):

            # need_col = ['process_review', 'Overall', 'value', 'room', 'location', 'clean', 'stuff', 'service', 'business']
            content = data[:, 0][i]
            tokens = [tok.replace('\n', '').lower() for tok in content.split(' ')]
            f = lambda w: (self.w2v_model.w2i[w] if w in self.w2v_model.w2i else self.w2v_model.w2i['<unk>'])
            w2v_id = [f(word) for word in tokens]
            # padding
            w2v_id = w2v_id[:max_l] if len(w2v_id) > max_l else w2v_id + [self.w2v_model.w2i['<pad>']] * (
                        max_l - len(w2v_id))
            label = data[:, 1][i]

            token_ids = self.tokenizer.encode(tokens)
            seq_len = len(token_ids)
            bert_max_l = max_l + 2
            if len(token_ids) < bert_max_l:
                mask = [1] * len(token_ids) + [0] * (bert_max_l - len(token_ids))
                token_ids += ([0] * (bert_max_l - len(token_ids)))
            else:
                mask = [1] * bert_max_l
                token_ids = token_ids[:bert_max_l - 1]
                token_ids += [self.tokenizer.convert_tokens_to_ids('[SEP]')]
                seq_len = bert_max_l

            w2v_ids.append(w2v_id)
            bert_ids.append(token_ids)
            labels.append(int(label))
            seq_lens.append(seq_len)
            masks.append(mask)

        w2v_ids = torch.tensor(w2v_ids)
        bert_ids = torch.tensor(bert_ids)
        labels = torch.tensor(labels)
        asp_info = torch.tensor(data[:, 2:].tolist())
        seq_lens = torch.tensor(seq_lens)
        masks = torch.tensor(masks)

        return w2v_ids, bert_ids, labels, asp_info, seq_lens, masks


    def get_dataset(self, max_l=200, duplicate=False):
        train_set = Data.TensorDataset(*self.preprocess(self.train_data, max_l, duplicate=duplicate))
        dev_set = Data.TensorDataset(*self.preprocess(self.dev_data, max_l, duplicate=duplicate))
        test_set = Data.TensorDataset(*self.preprocess(self.test_data, max_l, duplicate=duplicate))

        return self.w2v_model, train_set, dev_set, test_set

    def get_dataset_bert(self):
        train_set = Data.TensorDataset(*self.preprocess_bert(self.train_data))
        dev_set = Data.TensorDataset(*self.preprocess_bert(self.dev_data))
        test_set = Data.TensorDataset(*self.preprocess_bert(self.test_data))

        return self.w2v_model, train_set, dev_set, test_set


class GS_PreProcess():
    def __init__(self, sample=False, bert=False, duplicate=False, end2end=False):
        if duplicate:
            df = pd.read_excel(r"./data/processed/GreatSchools_for_super.xlsx")
            train_df = df[df['split'] == 'train']
            dev_df = df[df['split'] == 'valid']
            test_df = df[df['split'] == 'test']
            if sample:
                train_df = train_df.head(500)
                dev_df = dev_df.head(500)
                test_df = test_df.head(500)
            need_col = ['process_review', 'asp', 'asp_senti']
            self.train_data = train_df[need_col].values
            self.dev_data = dev_df[need_col].values
            self.test_data = test_df[need_col].values
        else:
            df = pd.read_excel(r"./data/processed/GreatSchools.xlsx")
            train_df = df[df['split'] == 'train']
            dev_df = df[df['split'] == 'valid']
            test_df = df[df['split'] == 'test']
            if sample:
                train_df = train_df.head(500)
                dev_df = dev_df.head(500)
                test_df = test_df.head(500)
            if end2end:
                col = ['Bullying policy', 'Character', 'LD Support', 'Leadership', 'Teachers']
                train_df[col] = train_df[col] + 2
                dev_df[col] = dev_df[col] + 2
                test_df[col] = test_df[col] + 2

            need_col = ['process_review', 'sentiment', 'Bullying policy', 'Character', 'LD Support', 'Leadership', 'Teachers']
            self.train_data = train_df[need_col].values
            self.dev_data = dev_df[need_col].values
            self.test_data = test_df[need_col].values

        sentences = train_df['process_review'].values
        w2v_path = './model_params/GS_comments_20asp.txt.w2v'
        self.w2v_model = word2vec(sentences)
        self.w2v_model.embed(w2v_path, d_embed=200, min_count=5)
        self.w2v_model.aspect(n_aspects=20)
        x = (self.w2v_model.n_vocab, self.w2v_model.d_embed, self.w2v_model.n_aspects)
        print('N_vocab: %d | D_embed: %d | N_aspects: %d' % x)

        if bert:
            self.tokenizer = BertTokenizer.from_pretrained('./model_params/bert/')
            self.bert_model = BertModel.from_pretrained("./model_params/bert")

            new_toks = []
            for t in self.w2v_model.i2w.values():
                if self.tokenizer.convert_tokens_to_ids(t) == self.tokenizer.unk_token_id:
                    new_toks.append(t)

            num_added_toks = self.tokenizer.add_tokens(new_toks)
            print('We have added', num_added_toks, 'tokens')
            self.bert_model.resize_token_embeddings(len(self.tokenizer))
            # save
            self.tokenizer.save_pretrained("./model_params/new_bert/")
            self.bert_model.save_pretrained("./model_params/new_bert/")


    def get_tokenized(self, data, duplicate=False):
        # data: list of ['processed_text', 'level', 'asp_info']
        if duplicate:
            return [tokenizer(review) for review, _, _ in data]
        else:
            return [tokenizer(review) for review, _, _, _, _, _, _ in data]

    def preprocess(self, data, max_l=200, duplicate=False):
        def pad(x):
            return x[:max_l] if len(x) > max_l else x + [self.w2v_model.w2i['<pad>']] * (max_l - len(x))
        f = lambda w: (self.w2v_model.w2i[w] if w in self.w2v_model.w2i else self.w2v_model.w2i['<unk>'])
        if duplicate:
            tokenized_data = self.get_tokenized(data, duplicate)
            features = torch.tensor([pad([f(word) for word in words]) for words in tokenized_data])
            orders = torch.tensor([o for _, o, _ in data])
            asp_sentis = torch.tensor([s for _, _, s in data])
            return features, orders, asp_sentis
        else:
            tokenized_data = self.get_tokenized(data)
            features = torch.tensor([pad([f(word) for word in words]) for words in tokenized_data])
            labels = torch.tensor([score for _, score, _, _, _, _, _ in data])
            asp_sentis = torch.tensor([[B, C, Ld, Le, T] for _, _, B, C, Ld, Le, T in data])
            return features, labels, asp_sentis

    def get_dataset(self, max_l=200, duplicate=False):
        train_set = Data.TensorDataset(*self.preprocess(self.train_data, max_l, duplicate=duplicate))
        dev_set = Data.TensorDataset(*self.preprocess(self.dev_data, max_l, duplicate=duplicate))
        test_set = Data.TensorDataset(*self.preprocess(self.test_data, max_l, duplicate=duplicate))

        return self.w2v_model, train_set, dev_set, test_set
