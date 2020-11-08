import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from modules import BasicModule as BM
from modules import cnn_module
from modules.Cluster import model_pred
from modules.metric_loss import MetricLoss
from utils import tool_box as tl
import config as cnf


class PCnn(BM.BasicModel):
    def __init__(self, word_vec_mat, max_len=120, pos_emb_dim=5, dropout=0.2, out_dim=64,save_dir='/home/wyt'):
        super(PCnn, self).__init__(save_dir)
        # if not random_init:
        #     seed_torch()
        self.batch_shape = (3, max_len)
        # defining layers
        dict_shape = word_vec_mat.shape
        self.word_emb = cnn_module.Embedding_word(dict_shape[0], dict_shape[1], weights=word_vec_mat,
                                                  requires_grad=True)
        # default trainable.
        self.pos1_emb = nn.Embedding(max_len * 2, pos_emb_dim)
        self.pos2_emb = nn.Embedding(max_len * 2, pos_emb_dim)
        self.drop = nn.Dropout(p=dropout)

        cnn_input_shape = (max_len, dict_shape[1] + 2 * pos_emb_dim)
        self.convnet = cnn_module.CNN(cnn_input_shape, out_dim)

        self.ml = MetricLoss.metric_loss()

    def forward(self, batch_input, perturbation=None):
        pos1 = batch_input[:, 0, :]
        pos2 = batch_input[:, 1, :]
        word = batch_input[:, 2, :]

        pos1_emb = self.pos1_emb(pos1)
        pos2_emb = self.pos2_emb(pos2)
        word_emb = self.word_emb(word)

        drop = self.drop(word_emb)
        if perturbation is not None:
            drop += perturbation

        cnn_input = torch.cat([drop, pos1_emb, pos2_emb], -1)
        cnn_input = cnn_input.permute([0, 2, 1])  # [B, embedding, max_len]
        encoded = self.convnet(cnn_input)

        return word_emb, encoded

    def forward_norm(self, batch_input, pertubation=None):
        word_emb, encoded = self.forward(batch_input, pertubation)
        encoded = self.norm(encoded)
        return word_emb, encoded

    def norm(self, encoded):
        return F.normalize(encoded, p=2, dim=1)

    def set_embedding_weight(self, weight):
        self.word_emb.word_embedding.weight.data[:-2].copy_(torch.from_numpy(weight))
        # torch.nn.init.xavier_uniform_(self.word_emb.word_embedding.weight.data[-2])

    # def get_vocab_size(self):
    #     vocab,dim=self.word_emb.word_embedding.weight.data.shape
    #     return (vocab-2,dim)


class RSN_ori(PCnn):
    def __init__(self, word_vec_mat, max_len=120, pos_emb_dim=5, dropout=0.2, out_dim=64, learning_rate=1e-4,save_dir='/home/wyt'):
        super(RSN_ori, self).__init__(word_vec_mat, max_len, pos_emb_dim, dropout, out_dim,save_dir)

        self.classifier = nn.Linear(out_dim, 1)

        self.lr = learning_rate
        params = []
        params += [{'params': self.word_emb.parameters(), 'lr': self.lr}]
        params += [{'params': self.pos1_emb.parameters(), 'lr': self.lr}]
        params += [{'params': self.pos2_emb.parameters(), 'lr': self.lr}]
        params += [{'params': list(self.convnet.conv1.parameters())[0], 'weight_decay': 2e-4, 'lr': self.lr}]
        # params += [{'params': list(self.convnet.conv1.parameters())[0]}]
        params += [{'params': list(self.convnet.conv1.parameters())[1], 'lr': self.lr}]

        params += [{'params': list(self.convnet.linear.parameters())[0], 'weight_decay': 1e-3, 'lr': self.lr}]
        # params += [{'params': list(self.convnet.linear.parameters())[0]}]
        params += [{'params': list(self.convnet.linear.parameters())[1], 'lr': self.lr}]
        params += [{'params': self.classifier.parameters(), 'lr': self.lr}]

        # import torchsummary
        # torchsummary
        self.optimizer = optim.Adam(params)

    def siamese_forward(self, left_input, right_input, left_perturbation=None, right_perturbation=None):
        # left_input/right_input:[batch * 3 * maxlen]
        left_word_emb, encoded_l = self.forward(left_input, left_perturbation)
        right_word_emb, encoded_r = self.forward(right_input, right_perturbation)
        both = torch.abs(encoded_l - encoded_r)
        # if not self.train_loss_type.startswith("triplet"):
        #     prediction = self.classifier(both).reshape(-1,1)
        #     # [batch * 1]
        #     prediction = F.sigmoid(prediction)
        # else:
        #     encoded_r = self.norm(encoded_r)
        #     encoded_l = self.norm(encoded_l)
        #     distances_squared = torch.sum(torch.pow(encoded_r - encoded_l, 2), dim=1)
        #     if self.squared:
        #         prediction = distances_squared
        #     else:
        #         prediction = distances_squared.sqrt()
        prediction = self.classifier(both).reshape(-1, 1)
        # [batch * 1]
        prediction = F.sigmoid(prediction)

        return prediction, left_word_emb, right_word_emb, encoded_l, encoded_r

    def train_self(self, opt, dataloader_trainset, dataloader_testset=None, batch_size=None):
        # if batch_size is None:
        #     batch_size = self.batch_size
        assert batch_size is not None
        self.train()

        # start_time = time.time()
        data_left, data_right, data_label = dataloader_trainset.next_batch(batch_size, same_ratio=opt.same_ratio)
        data_left, _ = tl.cudafy(data_left)
        data_right, _ = tl.cudafy(data_right)
        data_label, _ = tl.cudafy(data_label)
        prediction, left_word_emb, right_word_emb, encoded_l, encoded_r = self.siamese_forward(data_left, data_right)
        if opt.train_loss_type == "cross":
            loss = self.ml.get_cross_entropy_loss(prediction, labels=data_label)
        elif opt.train_loss_type == "cross_denoise":
            loss = self.ml.get_cross_entropy_loss(prediction, labels=data_label)
            loss += self.ml.get_cond_loss(prediction) * opt.p_denoise
        elif opt.train_loss_type == "v_adv":
            loss = self.ml.get_cross_entropy_loss(prediction, labels=data_label)
            loss += self.ml.get_v_adv_loss(self, data_left, data_right, opt.p_mult) * opt.lambda_s  # 可不可以把自己传进去呢？能不能传回来
        elif opt.train_loss_type == "v_adv_denoise":
            loss = self.ml.get_cross_entropy_loss(prediction, labels=data_label)
            loss += self.ml.get_v_adv_loss(self, data_left, data_right, opt.p_mult) * opt.lambda_s
            loss += self.ml.get_cond_loss(prediction) * opt.p_denoise
        else:
            raise NotImplementedError()

        # end_time = time.time()
        # print("Running time:", end_time - start_time, " seconds")

        # self.back_propagation(loss)
        self.optimizer.zero_grad()

        # print(self.pos1_emb.weight.detach().numpy())
        loss.backward()
        # no gradient for 'blank' word
        self.word_emb.word_embedding.weight.grad[-1] = 0
        self.optimizer.step()
        # end_time = time.time()
        # print("Running time:", end_time - start_time, " seconds")
        return loss

    def pred_X(self, data_left, data_right):
        self.eval()
        if isinstance(data_right, np.ndarray):
            data_right, _ = tl.cudafy(torch.from_numpy(np.array(data_right, dtype=np.int64)))
        if isinstance(data_left, np.ndarray):
            data_left, _ = tl.cudafy(torch.from_numpy(np.array(data_left, dtype=np.int64)))
        prediction, _l, _r, encoded_l, encoded_r = self.siamese_forward(data_left, data_right)
        return prediction, encoded_l, encoded_r


class RSN_FL(PCnn):
    def __init__(self,word_vec_mat,opt):
        max_len = opt.max_len
        pos_emb_dim = opt.pos_emb_dim
        dropout = opt.drop_out
        out_dim = opt.embedding_dim
        learning_rate = opt.lr[opt.lr_chose]
        save_dir = opt.save_dir

        super(RSN_FL, self).__init__(word_vec_mat, max_len, pos_emb_dim, dropout, out_dim,save_dir)

        self.squared = opt.squared
        self.lr = learning_rate
        params = []
        params += [{'params': self.word_emb.parameters(), 'lr': self.lr}]
        params += [{'params': self.pos1_emb.parameters(), 'lr': self.lr}]
        params += [{'params': self.pos2_emb.parameters(), 'lr': self.lr}]
        params += [{'params': list(self.convnet.conv1.parameters())[0], 'weight_decay': 2e-4, 'lr': self.lr}]
        # params += [{'params': list(self.convnet.conv1.parameters())[0]}]
        params += [{'params': list(self.convnet.conv1.parameters())[1], 'lr': self.lr}]

        params += [{'params': list(self.convnet.linear.parameters())[0], 'weight_decay': 1e-3, 'lr': self.lr}]
        # params += [{'params': list(self.convnet.linear.parameters())[0]}]
        params += [{'params': list(self.convnet.linear.parameters())[1], 'lr': self.lr}]

        # import torchsummary
        # torchsummary
        if opt.alpha is not None:
            self.optimizer = optim.RMSprop(params, alpha=opt.alpha)
        else:
            self.optimizer = optim.RMSprop(params)
        # torch.optim.RMSprop(net_RMSprop.parameters(), alpha=0.9)

    def pred_vector(self, data, opt):
        self.eval()
        if not isinstance(data, torch.Tensor):
            data, _ = tl.cudafy(torch.from_numpy(np.array(data, dtype=np.int64)))
        if opt.train_loss_type.startswith('Siamese'):
            _, vectors = self.forward(data)
        else:
            _, vectors = self.forward_norm(data)

        return vectors

    def pred_X(self, data_left, data_right):
        self.eval()
        if isinstance(data_right, np.ndarray):
            data_right, _ = tl.cudafy(torch.from_numpy(np.array(data_right, dtype=np.int64)))
        if isinstance(data_left, np.ndarray):
            data_left, _ = tl.cudafy(torch.from_numpy(np.array(data_left, dtype=np.int64)))
        _,vector_l = self.forward_norm(data_left)
        _, vector_r = self.forward_norm(data_right)

        distances_squared = torch.sum(torch.pow(vector_l - vector_r, 2), dim=1)
        if not self.squared:
            prediction = distances_squared.sqrt()
            # the euclidean dist between two normalized vector is in [0,2]
            rns = 1 - prediction / 2.0
        else:
            # the euclidean dist(squared) between two normalized vector is in [0,4]
            prediction = distances_squared
            rns = 1 - prediction / 4.0
        # prediction, _l, _r, encoded_l, encoded_r = self.siamese_forward(data_left, data_right)
        return rns, vector_l, vector_r

    def cos_smi(self, data_left, data_right):
        self.eval()
        if isinstance(data_right, np.ndarray):
            data_right, _ = tl.cudafy(torch.from_numpy(np.array(data_right, dtype=np.int64)))
        if isinstance(data_left, np.ndarray):
            data_left, _ = tl.cudafy(torch.from_numpy(np.array(data_left, dtype=np.int64)))
        _, vector_l = self.forward_norm(data_left)
        _, vector_r = self.forward_norm(data_right)

        length_l = torch.sum(torch.pow(vector_l,2),dim=1).sqrt()
        length_r = torch.sum(torch.pow(vector_r,2),dim=1).sqrt()

        rns = torch.sum(torch.mul(vector_l,vector_r),dim=1) / torch.mul(length_l,length_r).float()

        return rns, vector_l, vector_r

    def train_self(self, opt, dataloader_train, loss_list=None, loger=None, batch_chose=0,
                   dataloader_test = None, global_step=None,temp_epoch=1):
        batch_size = opt.batch_size[batch_chose]
        class_num_ratio = opt.class_num_ratio[batch_chose]
        # print(opt.class_num_ratio)
        # print(batch_chose)
        # print(opt.class_num_ratio[batch_chose])
        assert batch_size is not None
        self.train()

        # start_time=time.time()
        batch_data, batch_label, cluster_label = dataloader_train.next_batch_cluster(batch_size, class_num_ratio,
                                                                                     opt.batch_shuffle,opt.inclass_augment)
        batch_data, _ = tl.cudafy(batch_data)
        batch_label, _ = tl.cudafy(batch_label)

        wordembed, features = self.forward_norm(batch_data)
        loss = self.ml.cluster_loss(opt, batch_data,features, wordembed, batch_label, global_step,temp_epoch,self.forward_norm)

        if isinstance(loss, torch.Tensor):
            self.optimizer.zero_grad()
            loss.backward()
            self.word_emb.word_embedding.weight.grad[-1] = 0
            self.optimizer.step()
            loss_list.append(loss.item())

        if opt.whether_visualize and global_step == 1:
            loger.add_graph(self, input_to_model=batch_data)

        return loss_list

    def eval_self(self,val_data,val_data_labels,opt,global_step):
        self.eval()
        features = model_pred(val_data,self.pred_vector,opt)
        loss = self.ml.cluster_loss(opt, torch.from_numpy(features), torch.LongTensor(val_data_labels), global_step)
        return loss.item()

    def lr_decay(self, opt: cnf.Option):
        if opt.lr_chose < len(opt.lr) - 1:
            opt.lr_decay_num += 1
            opt.lr_chose += 1
            self.lr = opt.lr[opt.lr_chose]
            print("*** learning rate decay to {} ***".format(self.lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr


if __name__ == "__main__":
    word = torch.randn(114042, 50)
    # rp=Representation(word)
    # batch=torch.LongTensor(128, 3, 120).clamp(1, 200)
    # l=torch.LongTensor(128, 3, 120).clamp(1, 200)
    # r = torch.LongTensor(128, 3, 120).clamp(1, 200)
    # emd,enc=rp.forward(batch)
    #
    # print("embedding:{}".format(emd))
    # print("encoding:{}".format(enc))

    # rs = RSN_ori(word)
    # print(rs)
    # prediction, left_word_emb, right_word_emb, encoded_l, encoded_r=rs.siamese_forward(l,r)
    #
    # print("pre:{}".format(prediction))
    # print("emd_l:{}".format(left_word_emb))
    # print("emd_r:{}".format(right_word_emb))
    # print("enc_l:{}".format(encoded_l))
    # print("enc_r:{}".format(encoded_r))

    # rs.save_model("RSN")
    # rs.load_model("RSN_best.pt")

    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # RSN = tl.cudafy(RSN_ori(word))[0]
    # opt = cnf.Option()
    # dataloader_test = dataloader(opt.train_data_file, opt.wordvec_file)
    # RSN.train_self(opt, dataloader_test, batch_size=128, same_ratio=0.06)
    import warnings

    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    random_init = np.random.randn(114042, 50)
    opt = cnf.Option()
    rsn_fl = RSN_FL(random_init, opt)


