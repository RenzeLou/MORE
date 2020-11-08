import warnings
from pathlib import Path

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
        self.dataset="nyt"
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


        self.similarity_file = None # './data/support_files/trainset_similarity.pkl'
        self.same_level_pair_file = None  # './data/support_files/same_level_pair.json'
        self.gt_hierarchy_file = None # './data/support_files/all_structure.json'

        self.load_model_name = None  # 'RSN_FL_best.pt'  # file_name under checkpoints------------------------------------------
        self.save_model_name = 'RRL_nyt'
        self.save_dir = str(Path().absolute())

        self.lr = [0.0003]
        self.lr_chose = 0
        self.batch_size = [60,60,60,60]
        self.class_num_ratio = [0.5, 0.5, 0.5,0.5]
        # self.batch_size = [240, 240, 240, 240]
        # self.class_num_ratio = [0.1, 0.1, 0.1, 0.1]
        self.batch_num = [2000,2000,2000,2000]
        self.epoch_num = 4
        self.squared = False
        self.alpha = None
        self.lr_decay_num = 0  # real decay successfully num
        self.lr_decay_record = []  # global step when it go to decay
        self.batch_shuffle = False
        self.seed = 42
        self.early_stop=None  # None，if no early stop
        self.whether_visualize = True
        self.record_test = False
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
        self.K_num=16
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

    def parse(self,kwargs):
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has no attribute %s" %k)
            setattr(self,k.v)

        print("user config:")
        for k,v in self.__class__.__dict__.items():
            if not k.startswith("__"):
                print(k,getattr(self,k))
