import os
import numpy as np
import time
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, \
    v_measure_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
import torch.distributions as dist

from utils import tool_box as tl
from config import Option

NEAR_0 = 1e-10

class siamese(object):
    def __init__(self):
        super(siamese, self).__init__()

    def get_cross_entropy_loss(self, prediction, labels):
        labels_float = labels.float()
        loss = -torch.mean(
            labels_float * torch.log(prediction + NEAR_0) + (1 - labels_float) * torch.log(1 - prediction + NEAR_0))
        return loss

    # nn.CrossEntropyLoss
    def get_cond_loss(self, prediction):
        cond_loss = -torch.mean(
            prediction * torch.log(prediction + NEAR_0) + (1 - prediction) * torch.log(1 - prediction + NEAR_0))

        return cond_loss

    # original vat loss
    def get_v_adv_loss(self, model: nn.Module, ul_left_input, ul_right_input, p_mult, power_iterations=1):
        bernoulli = dist.Bernoulli
        prob, left_word_emb, right_word_emb = model.siamese_forward(ul_left_input, ul_right_input)[0:3]
        prob = prob.clamp(min=1e-7, max=1. - 1e-7)
        prob_dist = bernoulli(probs=prob)
        # generate virtual adversarial perturbation
        left_d, _ = tl.cudafy(torch.FloatTensor(left_word_emb.shape).uniform_(0, 1))
        right_d, _ = tl.cudafy(torch.FloatTensor(right_word_emb.shape).uniform_(0, 1))
        left_d.requires_grad, right_d.requires_grad = True, True
        # prob_dist.requires_grad = True
        # kl_divergence
        for _ in range(power_iterations):
            left_d = (0.02) * F.normalize(left_d, p=2, dim=1)
            right_d = (0.02) * F.normalize(right_d, p=2, dim=1)
            # d1 = dist.Categorical(a)
            # d2 = dist.Categorical(torch.ones(5))
            p_prob = model.siamese_forward(ul_left_input, ul_right_input, left_d, right_d)[0]
            p_prob = p_prob.clamp(min=1e-7, max=1. - 1e-7)
            # torch.distribution
            try:
                kl = dist.kl_divergence(prob_dist, bernoulli(probs=p_prob))
            except:
                wait = True
            left_gradient, right_gradient = torch.autograd.grad(kl.sum(), [left_d, right_d], retain_graph=True)
            left_d = left_gradient.detach()
            right_d = right_gradient.detach()
        left_d = p_mult * F.normalize(left_d, p=2, dim=1)
        right_d = p_mult * F.normalize(right_d, p=2, dim=1)
        # virtual adversarial loss
        p_prob = model.siamese_forward(ul_left_input, ul_right_input, left_d, right_d)[0].clamp(min=1e-7, max=1. - 1e-7)
        v_adv_losses = dist.kl_divergence(prob_dist, bernoulli(probs=p_prob))

        return torch.mean(v_adv_losses)