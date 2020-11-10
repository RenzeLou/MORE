import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import BasicModule as BM
from modules import cnn_module
from modules.Cluster import model_pred
from modules.metric_loss import MetricLoss
from modules.metric_loss.MetricLoss import pairwise_distance
from utils import tool_box as tl
from torch import optim


class PCnn(BM.BasicModel):
    def __init__(self, word_vec_mat, max_len=120, pos_emb_dim=5, dropout=0.2, out_dim=64, save_dir='/home/wyt'):
        super(PCnn, self).__init__(save_dir)
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


class MORE(PCnn):
    def __init__(self, word_vec_mat, opt):
        max_len = opt.max_len
        pos_emb_dim = opt.pos_emb_dim
        dropout = opt.drop_out
        out_dim = opt.embedding_dim
        learning_rate = opt.lr[opt.lr_chose]
        save_dir = opt.save_dir

        super(MORE, self).__init__(word_vec_mat, max_len, pos_emb_dim, dropout, out_dim, save_dir)

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
        _, vector_l = self.forward_norm(data_left)
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

        length_l = torch.sum(torch.pow(vector_l, 2), dim=1).sqrt()
        length_r = torch.sum(torch.pow(vector_r, 2), dim=1).sqrt()

        rns = torch.sum(torch.mul(vector_l, vector_r), dim=1) / torch.mul(length_l, length_r).float()

        return rns, vector_l, vector_r

    def train_self(self, opt, dataloader_train, loss_list=None, loger=None, batch_chose=0,
                   global_step=None, temp_epoch=1):
        batch_size = opt.batch_size[batch_chose]
        class_num_ratio = opt.class_num_ratio[batch_chose]

        assert batch_size is not None
        self.train()

        batch_data, batch_label, cluster_label = dataloader_train.next_batch_cluster(batch_size, class_num_ratio,
                                                                                     opt.batch_shuffle,
                                                                                     opt.inclass_augment)
        batch_data, _ = tl.cudafy(batch_data)
        batch_label, _ = tl.cudafy(batch_label)

        wordembed, features = self.forward_norm(batch_data)  # [batch_size,embedding_dim]

        if opt.VAT != 0 and temp_epoch >= opt.warm_up:
            # add VAT
            total_loss = 0.0
            labels = batch_label
            margin = opt.margin
            alpha = opt.alpha_rank
            tval = opt.temp_neg
            encode_ori = features

            dist_mat = pairwise_distance(encode_ori, opt.squared)  # [batch,batch]
            ori_distribution = torch.FloatTensor([]).cuda()
            ori_distribution.requires_grad = True

            # compute score_ori and RLL
            for achor in range(dist_mat.shape[0]):
                is_pos = labels.eq(labels[achor])
                is_pos[achor] = 0
                is_neg = labels.ne(labels[achor])

                dist_ap = dist_mat[achor][is_pos]
                dist_an = dist_mat[achor][is_neg]

                ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
                ap_pos_num = ap_is_pos.size(0) + 1e-5
                ap_pos_val_sum = torch.sum(ap_is_pos)
                loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

                an_is_pos = torch.lt(dist_an, alpha)
                an_less_alpha = dist_an[an_is_pos]
                an_weight = torch.exp(tval * (-1 * an_less_alpha + alpha))
                an_weight_sum = torch.sum(an_weight) + 1e-5
                an_dist_lm = alpha - an_less_alpha
                an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight))
                loss_an = torch.div(an_ln_sum, an_weight_sum)
                total_loss = total_loss + loss_ap + loss_an

            disturb, _ = tl.cudafy(torch.FloatTensor(wordembed.shape).uniform_(0, 1))

            # kl_divergence
            for _ in range(opt.power_iterations):
                disturb.requires_grad = True
                disturb = (opt.p_mult) * F.normalize(disturb, p=2, dim=1)
                _, encode_disturb = self.forward_norm(batch_data, disturb)
                dist_el = torch.sum(torch.pow(encode_ori - encode_disturb, 2), dim=1).sqrt()
                diff = (dist_el / 2.0).clamp(0, 1.0 - 1e-7)
                disturb_gradient = torch.autograd.grad(diff.sum(), disturb, retain_graph=True)[0]
                disturb = disturb_gradient.detach()

            disturb = opt.p_mult * F.normalize(disturb, p=2, dim=1)

            # virtual adversarial loss
            _, encode_final = self.forward_norm(batch_data, disturb)

            # compute pair wise use the new embedding
            final_distribution = torch.FloatTensor([]).cuda()
            final_distribution.requires_grad = True
            dist_el = torch.sum(torch.pow(encode_ori - encode_final, 2), dim=1).sqrt()
            diff = (dist_el / 2.0).clamp(0, 1.0 - 1e-7)
            v_adv_losses = torch.mean(diff)
            loss = total_loss * 1.0 / dist_mat.size(0) + v_adv_losses * opt.lambda_V
            assert torch.mean(v_adv_losses).item() > 0.0
        else:
            self.ml = MetricLoss.metric_loss()
            loss = self.ml.cluster_loss(opt, features, batch_label, global_step)

        if isinstance(loss, torch.Tensor):
            self.optimizer.zero_grad()
            loss.backward()
            self.word_emb.word_embedding.weight.grad[-1] = 0
            self.optimizer.step()
            loss_list.append(loss.item())

        if opt.whether_visualize and global_step == 1:
            loger.add_graph(self, input_to_model=batch_data)

        return loss_list
        # print("loss:",loss)

    def eval_self(self, val_data, val_data_labels, opt, global_step):
        self.eval()
        features = model_pred(val_data, self.pred_vector, opt)
        loss = self.ml.cluster_loss(opt, torch.from_numpy(features), torch.LongTensor(val_data_labels), global_step)
        return loss.item()

    def lr_decay(self, opt):
        if opt.lr_chose < len(opt.lr) - 1:
            opt.lr_decay_num += 1
            opt.lr_chose += 1
            self.lr = opt.lr[opt.lr_chose]
            print("*** learning rate decay to {} ***".format(self.lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr


class MORE_BERT(BM.BasicModel):
    def __init__(self, opt):
        super(MORE_BERT, self).__init__(opt.save_dir)
        self.opt = opt

        dropout = opt.drop_out

        self.Bert = opt.model

        self.drop = nn.Dropout(p=dropout)

        self.ml = MetricLoss.metric_loss()

        self.lr = opt.lr[opt.lr_chose]
        self.lr_linear = opt.lr_linear

        self.Bert_model = opt.Bert_model

        self.FFL = nn.Linear(1536, 64)
        self.squared = opt.squared

        # Setting all parameters grad True
        for param in self.Bert.parameters():
            param.requires_grad = False

        for param in self.Bert_model.parameters():
            param.requires_grad = True

        # check how many parameters require grad
        print('# params require grad', len(list(filter(lambda p: p.requires_grad, self.Bert_model.parameters()))))

        param = []
        param += [{'params': filter(lambda p: p.requires_grad, self.Bert_model.parameters()), 'lr': self.lr}]
        param += [{'params': list(self.FFL.parameters())[0], 'weight_decay': 1e-3, 'lr': self.lr_linear}]
        param += [{'params': list(self.FFL.parameters())[1], 'lr': self.lr_linear}]

        self.optimizer = optim.Adam(param)
        self.lr_decays = 0

    def pred_vector(self, data, labels, opt):
        self.eval()

        if opt.train_loss_type.startswith('Siamese'):
            hidden, _, _ = self.bert_forward(data, labels)  # forward
            vectors = self.FFL(hidden)
            # self.linear(vectors)
        else:
            # batch_size, sequence_size, 16, 128
            hidden, _, _ = self.bert_forward(data, labels)  # forward norm
            features = self.FFL(hidden)
            vectors = self.norm_layer(features)

        return vectors

    def train_self(self, opt, dataloader_train, loss_list=None, loger=None, batch_chose=0,
                   global_step=None, temp_epoch=0, chose_decay=False):
        # batch size 60, num_ratio 0.5
        batch_size = opt.batch_size[batch_chose]
        class_num_ratio = opt.class_num_ratio[batch_chose]
        assert batch_size is not None
        self.train()

        # learning rate decay
        if temp_epoch > 0 and self.lr > 1e-8 and chose_decay:
            print("lr decay to {}!".format(self.lr * 0.1))
            self.lr = self.lr * 0.1
            param = []
            param += [{'params': filter(lambda p: p.requires_grad, self.Bert_model.parameters()), 'lr': self.lr}]
            param += [{'params': list(self.FFL.parameters())[0], 'weight_decay': 1e-3, 'lr': self.lr_linear}]
            param += [{'params': list(self.FFL.parameters())[1], 'lr': self.lr_linear}]

            self.optimizer = optim.Adam(param).to(opt.device)

        # torch tensor batch_data [batch_size, sequence], batch_sentence is BERT input
        batch_data, batch_label, batch_sentence, cluster_label = dataloader_train.next_batch_cluster(batch_size,
                                                                                                     class_num_ratio,
                                                                                                     opt.batch_shuffle,
                                                                                                     opt.inclass_augment)

        # BERT forward with batch size of 8
        marker_temper_data, b_input_ids, batch_label = self.bert_forward(batch_sentence, batch_label)
        batch_label, _ = tl.cudafy(batch_label)

        # Go through fully connect layer and RELU, then norm layer
        features = self.FFL(marker_temper_data)
        features = self.norm_layer(features)

        # Loss calculation
        loss = self.ml.cluster_loss(opt, features, batch_label, global_step)  # check

        # Backward
        if isinstance(loss, torch.Tensor):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())

        if opt.whether_visualize and global_step == 1:
            try:
                loger.add_graph(self, input_to_model=batch_data)
            except:
                print("*tensorboard : add graph failed")

        return loss_list

    def bert_forward(self, batch_sentence, batch_label):
        opt = self.opt

        # tokenize and get all data
        b_input_ids, b_index, b_labels = opt.pre_processing(batch_sentence, batch_label)

        # retrieving the BERT dataloader
        dataloader_bert = opt.batch_dataset(b_input_ids, b_index, b_labels, opt.bert_batch_size)

        # marker_temper_data = []

        marker_temper_data = torch.FloatTensor([]).to(opt.device)
        marker_temper_data.requires_grad = True
        batch_label_process = torch.LongTensor([]).to(opt.device)

        # Forward whole batch
        for step, batch in enumerate(dataloader_bert):
            batch = tuple(t.to(opt.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_index, b_labels = batch
            batch_label_process = torch.cat((batch_label_process, b_labels))

            # fixing type error, numpy to tensor
            b_input_ids = b_input_ids.type(torch.LongTensor)
            b_index = b_index.type(torch.LongTensor)
            b_labels = b_labels.type(torch.LongTensor)

            b_input_ids = b_input_ids.to(opt.device)
            b_index = b_index.to(opt.device)
            b_labels = b_labels.to(opt.device)

            # feed in batch_data
            outputs = self.Bert_model(b_input_ids)
            last_hidden_states = outputs[0]
            logits = last_hidden_states

            # get hidden state of two entities
            for i in range(len(b_input_ids)):
                temp1 = logits[i, b_index[i][0]]
                temp2 = logits[i, b_index[i][1]]
                temp = torch.cat((temp1, temp2))
                temp = temp.unsqueeze(0)
                marker_temper_data = torch.cat((marker_temper_data, temp), dim=0)

        return marker_temper_data, b_input_ids, batch_label_process  # [240,1536]

    def eval_self(self, val_data, val_data_labels, opt, global_step):
        self.eval()
        features = model_pred(val_data, self.pred_vector, val_data_labels, opt)
        loss = self.ml.cluster_loss(opt, torch.from_numpy(features), torch.LongTensor(val_data_labels), global_step)
        return loss.item()

    def lr_decay(self, opt):
        if opt.lr_chose < len(opt.lr) - 1:
            opt.lr_decay_num += 1
            opt.lr_chose += 1
            self.lr = opt.lr[opt.lr_chose]
            print("*** learning rate decay to {} ***".format(self.lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def norm_layer(self, bert_input):
        encoded = self.norm(bert_input)
        return encoded

    def norm(self, encoded):
        return F.normalize(encoded, p=2, dim=1)


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

    # rs = more_ori(word)
    # print(rs)
    # prediction, left_word_emb, right_word_emb, encoded_l, encoded_r=rs.siamese_forward(l,r)
    #
    # print("pre:{}".format(prediction))
    # print("emd_l:{}".format(left_word_emb))
    # print("emd_r:{}".format(right_word_emb))
    # print("enc_l:{}".format(encoded_l))
    # print("enc_r:{}".format(encoded_r))

    # rs.save_model("more")
    # rs.load_model("more_best.pt")

    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # more = tl.cudafy(more_ori(word))[0]
    # opt = cnf.Option()
    # dataloader_test = dataloader(opt.train_data_file, opt.wordvec_file)
    # more.train_self(opt, dataloader_test, batch_size=128, same_ratio=0.06)
    # import warnings
    #
    # warnings.filterwarnings("ignore")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # random_init = np.random.randn(114042, 50)
    # opt = cnf.Option()
    # MORE = MORE(random_init, opt)
