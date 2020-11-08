import torch
import random
import numpy as np
import torch.nn.functional as F
from utils import tool_box as tl


def pairwise_distance(embeddings, squared=False):
    pairwise_distances_squared = torch.sum(embeddings ** 2, dim=1, keepdim=True) + \
                                 torch.sum(embeddings.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(embeddings, embeddings.t())

    error_mask = pairwise_distances_squared <= 0.0
    if squared:
        pairwise_distances = pairwise_distances_squared.clamp(min=0)
    else:
        pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    num_data = embeddings.shape[0]
    # Explicitly set diagonals to zero.
    if pairwise_distances.is_cuda:
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(tl.cudafy(torch.ones([num_data]))[0])
    else:
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(torch.ones([num_data]))

    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


def uninformative_loss(dist, x, boundary, margin, drop=None):
    # I attempt to utilize the uninformative points, but unfortunately it does not work, it will be left for future work
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
    '''
    The original code come from :
    https://github.com/Qidian213/Ranked_Person_ReID
    '''

    def __init__(self):
        super(rank_list, self).__init__()

    def compute(self, batch_data, features, wordembed, labels, opt, temp_epoch, forward_norm):
        # margin = opt.margin
        # alpha = opt.alpha_rank
        # tval = opt.temp_neg
        # landa = opt.landa
        #
        # assert len(dist_mat.size()) == 2
        # assert dist_mat.size(0) == dist_mat.size(1)
        # N = int(dist_mat.size(0) / 2 if opt.inclass_augment else dist_mat.size(0))
        # ori_label = copy.deepcopy(labels)
        # virtual_label = labels[N:]
        # labels = labels[0:N]
        #
        # total_loss = 0.0
        # loss_pp = 0.0
        # loss_nn = 0.0
        # for ind in range(N):
        #     is_pos = labels.eq(labels[ind])
        #     is_pos[ind] = 0
        #     is_neg = labels.ne(labels[ind])
        #
        #     dist_ap = dist_mat[ind][0:N][is_pos]
        #     dist_an = dist_mat[ind][0:N][is_neg]
        #
        #     ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
        #     ap_pos_num = ap_is_pos.size(0) + 1e-5
        #     ap_pos_val_sum = torch.sum(ap_is_pos)
        #     loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))
        #
        #     an_is_pos = torch.lt(dist_an, alpha)
        #     an_less_alpha = dist_an[an_is_pos]
        #     an_weight = torch.exp(tval * (-1 * an_less_alpha + alpha))
        #     an_weight_sum = torch.sum(an_weight) + 1e-5
        #     an_dist_lm = alpha - an_less_alpha
        #     an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight))
        #     loss_an = torch.div(an_ln_sum, an_weight_sum)
        #
        #     total_loss = total_loss + loss_ap + loss_an
        #     loss_nn += loss_an
        #     loss_pp += loss_ap
        #
        # if opt.inclass_augment:
        #     total_loss_virtual = 0.0
        #     loss_au = 0.0
        #     for ind in range(N):
        #         is_pos = ori_label.eq(virtual_label[ind])
        #         is_pos[ind] = 0
        #         dist_ap = dist_mat[N + ind][is_pos]
        #
        #         ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
        #         ap_pos_num = ap_is_pos.size(0) + 1e-5
        #         ap_pos_val_sum = torch.sum(ap_is_pos)
        #         loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))
        #
        #         total_loss_virtual += loss_ap
        #         loss_au += loss_ap
        #     total_loss += total_loss_virtual
        #
        # total_loss = total_loss * 1.0 / N
        #
        # return total_loss

        total_loss = 0.0
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
            pos_score = torch.mean(ap_is_pos / (2 - alpha + margin))
            ap_pos_num = ap_is_pos.size(0) + 1e-5
            ap_pos_val_sum = torch.sum(ap_is_pos)
            loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

            an_is_pos = torch.lt(dist_an, alpha)
            an_less_alpha = dist_an[an_is_pos]
            neg_score = torch.mean(torch.clamp((alpha - dist_an), min=0.0) / alpha)
            an_weight = torch.exp(tval * (-1 * an_less_alpha + alpha))
            an_weight_sum = torch.sum(an_weight) + 1e-5
            an_dist_lm = alpha - an_less_alpha
            an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight))
            loss_an = torch.div(an_ln_sum, an_weight_sum)

            total_loss = total_loss + loss_ap + loss_an

        if opt.VAT != 0 and temp_epoch >= opt.warm_up:
            # add VAT
            # print("add VAT")
            disturb, _ = tl.cudafy(torch.FloatTensor(wordembed.shape).uniform_(0, 1))
            for _ in range(opt.power_iterations):
                disturb.requires_grad = True
                disturb = (opt.p_mult) * F.normalize(disturb, p=2, dim=1)
                _, encode_disturb = forward_norm(batch_data, disturb)

                dist_el = torch.sum(torch.pow(encode_ori - encode_disturb, 2), dim=1).sqrt()

                diff = (dist_el / 2.0).clamp(0, 1.0 - 1e-7)

                disturb_gradient = torch.autograd.grad(diff.sum(), disturb, retain_graph=True)[0]

                disturb = disturb_gradient.detach()

            disturb = opt.p_mult * F.normalize(disturb, p=2, dim=1)
            # virtual adversarial loss
            _, encode_final = forward_norm(batch_data, disturb)
            # compute pair wise use the new embedding
            final_distribution = torch.FloatTensor([]).cuda()
            final_distribution.requires_grad = True

            dist_el = torch.sum(torch.pow(encode_ori - encode_final, 2), dim=1).sqrt()

            diff = (dist_el / 2.0).clamp(0, 1.0 - 1e-7)

            v_adv_losses = torch.mean(diff)

            loss = total_loss * 1.0 / dist_mat.size(0) + v_adv_losses * opt.lambda_V

            assert torch.mean(v_adv_losses).item() > 0.0
        else:
            loss = total_loss * 1.0 / dist_mat.size(0)

        return loss
