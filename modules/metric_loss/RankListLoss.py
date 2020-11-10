import torch
import random
import copy
import numpy as np


def uninformative_loss(dist, x, boundary, margin, drop=None):
    # During the experiments, I attempt to utilize the uninformative points to improve optimization, but unfortunately it doesn't work
    pos_dist = dist[x < boundary]
    neg_dist = dist[x > boundary]
    pos_x = x[x < boundary]
    neg_x = x[x > boundary]

    pos_weight = -torch.log(torch.div(pos_x, boundary))
    neg_weight = torch.log(torch.div(neg_x, boundary))
    all_weight = torch.sum(pos_weight) + torch.sum(neg_weight)

    pos_weight = torch.div(pos_weight, all_weight)
    neg_weight = torch.div(neg_weight, all_weight)

    if drop is not None:
        random_zeros_list_pos = random.sample(list(np.arange(len(pos_dist))), int(len(pos_dist) * drop))
        random_zeros_pos = torch.from_numpy(np.array(random_zeros_list_pos)).long()
        pos_dist[random_zeros_pos] = margin

        random_zeros_list_neg = random.sample(list(np.arange(len(neg_dist))), int(len(neg_dist) * drop))
        random_zeros_neg = torch.from_numpy(np.array(random_zeros_list_neg)).long()
        neg_dist[random_zeros_neg] = margin

    pos_loss = torch.sum(torch.mul(torch.abs(pos_dist - margin), pos_weight))
    neg_loss = torch.sum(torch.mul(torch.abs(margin - neg_dist), neg_weight))

    all_loss = pos_loss + neg_loss

    return all_loss


class rank_list(object):
    def __init__(self):
        super(rank_list, self).__init__()

    def compute(self, dist_mat, labels, opt, global_step):
        """
        The original class come from : https://github.com/Qidian213/Ranked_Person_ReID

        Args:
          dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
          labels: pytorch LongTensor, with shape [N]

        """

        margin = opt.margin
        alpha = opt.alpha_rank
        tval = opt.temp_neg
        landa = opt.landa

        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = int(dist_mat.size(0) / 2 if opt.inclass_augment else dist_mat.size(0))
        ori_label = copy.deepcopy(labels)
        virtual_label = labels[N:]
        labels = labels[0:N]

        total_loss = 0.0
        loss_pp = 0.0
        loss_nn = 0.0
        for ind in range(N):
            is_pos = labels.eq(labels[ind])
            is_pos[ind] = 0
            is_neg = labels.ne(labels[ind])

            dist_ap = dist_mat[ind][0:N][is_pos]
            dist_an = dist_mat[ind][0:N][is_neg]

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
            loss_nn += loss_an
            loss_pp += loss_ap

        # print("<<<<<<<<<<<loss_positive: {},loss_negative: {}".format(loss_pp,loss_nn))
        if opt.inclass_augment:
            total_loss_virtual = 0.0
            loss_au = 0.0
            for ind in range(N):
                is_pos = ori_label.eq(virtual_label[ind])
                is_pos[ind] = 0
                dist_ap = dist_mat[N + ind][is_pos]

                ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
                ap_pos_num = ap_is_pos.size(0) + 1e-5
                ap_pos_val_sum = torch.sum(ap_is_pos)
                loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

                total_loss_virtual += loss_ap
                loss_au += loss_ap
            total_loss += total_loss_virtual

        total_loss = total_loss * 1.0 / N

        return total_loss

    # margin = opt.margin  # 0.2
    # margin_un = opt.margin_un  # 0.4
    # alpha = opt.alpha_rank  # 1.0
    # alpha_un = opt.alpha_rank_un  # 1.2
    # tval = opt.temp_neg  # 10
    # tval_p = opt.temp_pos  # 0
    # landa = opt.landa  # 0.5
    #
    # un_landa = opt.uninfor_landa
    # un_drop = opt.uninfor_drop
    # # un_drop = opt.uninfor_drop if global_step > 4000 else None
    #
    # assert len(dist_mat.size()) == 2
    # assert dist_mat.size(0) == dist_mat.size(1)
    # N = dist_mat.size(0)
    #
    # total_loss = 0.0
    # for ind in range(N):
    #     is_pos = labels.eq(labels[ind])
    #     is_pos[ind] = 0
    #     is_neg = labels.ne(labels[ind])
    #
    #     dist_ap = dist_mat[ind][is_pos]
    #     dist_an = dist_mat[ind][is_neg]
    #     # informative sample loss compute--------------------------------------
    #     ap_is_pos = torch.gt(dist_ap, alpha - margin)  # >0.8
    #     ap_great_margin = dist_ap[ap_is_pos]
    #     ap_weight = torch.exp(tval_p * (ap_great_margin - alpha + margin))
    #     ap_weight_sum = torch.sum(ap_weight) + 1e-5
    #     ap_dist_gm = ap_great_margin - alpha + margin
    #     ap_gm_sum = torch.sum(torch.mul(ap_dist_gm, ap_weight))
    #     loss_ap = torch.div(ap_gm_sum, ap_weight_sum)
    #
    #     an_is_pos = torch.lt(dist_an, alpha)  # <1.0
    #     an_less_alpha = dist_an[an_is_pos]
    #     an_weight = torch.exp(tval * (-1 * an_less_alpha + alpha))
    #     an_weight_sum = torch.sum(an_weight) + 1e-5
    #     an_dist_lm = alpha - an_less_alpha
    #     an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight))
    #     loss_an = torch.div(an_ln_sum, an_weight_sum)
    #
    #     # uninformative sample loss compute---------------------------------------------
    #     # ap_is_un = torch.lt(dist_ap, alpha - margin)  # <0.8
    #     # ap_less_marign = dist_ap[ap_is_un]
    #     # x = alpha - margin - ap_less_marign
    #     # uninfor_margin_pos = alpha - margin_un  # 0.6
    #     # loss_ap_un = uninformative_loss(ap_less_marign,x,margin_un - margin,uninfor_margin_pos)
    #     #
    #     # an_is_un = torch.gt(dist_an,alpha)  # >1.0
    #     # an_great_alpha = dist_an[an_is_un]
    #     # x = an_great_alpha - alpha
    #     # uninfor_margin_neg = alpha_un  # 1.2
    #     # loss_an_un = uninformative_loss(an_great_alpha,x,alpha_un - alpha,uninfor_margin_neg,un_drop)
    #
    #     # total_loss = total_loss + (1 - landa) * (loss_ap + loss_ap_un) + landa * (loss_an + loss_an_un)
    #     total_loss = total_loss + loss_ap + loss_an
