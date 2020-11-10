import numpy as np
import warnings
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, \
    BertModel  # convert pretrain_bert to transformers to add special tokens


class Option(object):
    def __init__(self):
        super(Option, self).__init__()

        self.train_loss_type = "Rank List Loss"
        self.testset_loss_type = 'none'

        # self.data_type = 'trex'
        # self.train_data_file = './data/datasets/Trex/train-spo.json'
        # self.val_data_file = './data/datasets/Trex/dev-spo.json'
        # self.test_data_file = './data/datasets/Trex/test-spo.json'
        # self.wordvec_file = "./data/wordvec/word_vec.json"
        # self.rel2id_file = './data/datasets/Trex/rel2id.json'
        self.gpu = "0"
        self.dataset = "nyt"
        self.data_type = 'nyt_fb'
        self.train_data_file = '/home/wyt/lrz/data/datasets/nyt_fb/nyt_ori/nyt_all_train.json'
        self.val_data_file = '/home/wyt/lrz/data/datasets/nyt_fb/nyt_ori/nyt_all_dev.json'
        self.test_data_file = '/home/wyt/lrz/data/datasets/nyt_fb/nyt_ori/nyt_all_test.json'
        self.wordvec_file = "/home/wyt/lrz/data/wordvec/word_vec.json"
        self.rel2id_file = '/home/wyt/lrz/data/datasets/nyt_fb/nyt_ori/rel2id_cnn.json'

        # self.data_type = 'fewrel'
        # self.train_data_file = './data/datasets/fewrel_ori/fewrel80_train.json'
        # self.val_data_file = './data/datasets/fewrel_ori/fewrel80_test_train.json'
        # self.test_data_file = './data/datasets/fewrel_ori/fewrel80_test_test.json'
        # self.wordvec_file = "./data/wordvec/word_vec.json"
        # self.rel2id_file = './data/support_files/rel2id.json'

        self.similarity_file = None  # './data/support_files/trainset_similarity.pkl'
        self.same_level_pair_file = None  # './data/support_files/same_level_pair.json'
        self.gt_hierarchy_file = None  # './data/support_files/all_structure.json'

        self.load_model_name = None  # 'MORE_best.pt' , the file_name (under checkpoints)
        self.save_model_name = 'MORE'
        self.save_dir = str(Path().absolute())

        self.lr = [0.0003]
        self.lr_chose = 0
        self.batch_size = [60, 60, 60, 60]
        self.class_num_ratio = [0.5, 0.5, 0.5, 0.5]
        # self.batch_size = [240, 240, 240, 240]
        # self.class_num_ratio = [0.1, 0.1, 0.1, 0.1]
        self.batch_num = [2000, 2000, 2000, 2000]
        self.epoch_num = 4
        self.squared = False
        self.alpha = None
        self.lr_decay_num = 0  # real decay successfully num
        self.lr_decay_record = []  # global step when it go to decay
        self.batch_shuffle = False
        self.seed = 42
        self.early_stop = None  # None，if no early stop
        self.whether_visualize = True
        self.record_test = True
        self.embedding_dim = 64
        self.max_len = 120
        self.pos_emb_dim = 5
        self.drop_out = 0.2
        self.same_level_part = 200
        self.mask_same_level_epoch = 100
        self.random_init = False
        self.print_losses = False
        self.select_cluster = 'K-means'
        self.eval_num = 5
        self.K_num = 16
        self.band = 0.7784975910442384

        self.margin = 0.2
        self.margin_un = 0.4
        self.alpha_rank = 1.0
        self.alpha_rank_un = 1.2
        self.temp_neg = 10
        self.temp_pos = 0
        self.landa = 0.5
        self.VAT = 0.05
        self.p_mult = 0.02
        self.power_iterations = 1
        self.warm_up = 0
        self.lambda_V = 1
        self.inclass_augment = False

        self.uninfor_landa = 0.0
        self.uninfor_drop = 0.0

        self.BERT = False

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has no attribute %s" % k)
            setattr(self, k.v)

        print("user config:")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("__"):
                print(k, getattr(self, k))


class Bert_option(Option):
    def __init__(self, gpu):
        super(Bert_option, self).__init__()

        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        import torch
        import torch.nn as nn

        if torch.cuda.is_available():
            torch.cuda.set_device(int(gpu))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BERT = True

        print(">>>Note that you chose to use BERT for extractor!")

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower_case=True,
                                                       never_split=(
                                                           "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "<e1>",
                                                           "</e1>",
                                                           "<e2>", "</e2>"))

        # using pretrained bert model
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

        self.Bert_model = nn.Sequential(*list(self.model.children())[:-2])

        # add token to vocabulary  =======================================================
        special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        assert num_added_toks == 4
        self.model.resize_token_embeddings(len(self.tokenizer))

        e1_id = self.tokenizer.convert_tokens_to_ids('<e1>')
        e2_id = self.tokenizer.convert_tokens_to_ids('<e2>')
        print("e1:", e1_id, " e2:", e2_id)
        assert e1_id != e2_id != 1

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        count = count_parameters(self.Bert_model)
        print("model_num_parameters:", count)

        self.bert_batch_size = 16

        self.word_vec_mat = np.random.randn(114042, 50)

        # have some special training methods
        self.lr = [1e-5]
        self.lr_linear = 1e-4
        self.lr_chose = 0
        # self.batch_size = [60,60,60,60]
        # self.class_num_ratio = [0.5, 0.5, 0.5,0.5]
        self.batch_size = [40, 40, 40, 40]
        self.class_num_ratio = [0.5, 0.5, 0.5, 0.5]
        self.batch_num = [1000, 1000, 1000, 1000]

    def batch_dataset(self, inputs, indexs, labels, batch_size):
        data = TensorDataset(inputs, indexs, labels)
        # sampler = RandomSampler(data)
        sampler = SequentialSampler(data)
        train_dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return train_dataloader

    def pre_processing(self, sentence, sentence_label):
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentence]  # tokenization
        # reset the sentence index =========================================================
        sen_index = []
        for sen in tokenized_texts:
            e1_begin = sen.index('<e1>')
            e1_end = sen.index('</e1>')
            e2_begin = sen.index('<e2>')
            e2_end = sen.index('</e2>')

            sen_index.append([e1_begin, e2_begin])

        try:
            assert len(sen_index) == len(tokenized_texts)
            sentence_index = sen_index
        except:
            raise Exception("error!")
        MAX_LEN = 128
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        # Pad our input tokens
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        # Convert all of our data into torch tensors, the required datatype for our model

        import torch

        inputs = torch.tensor(input_ids)
        index = torch.tensor(sentence_index)
        labels = torch.tensor(sentence_label)
        masks = torch.tensor(attention_masks)

        return inputs, index, labels


if __name__ == '__main__':
    Bert_option()
