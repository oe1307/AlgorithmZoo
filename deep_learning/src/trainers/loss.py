import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcFace(nn.Module):
    def __init__(self, config):
        super(ArcFace, self).__init__()
        self.num_features = config.embedding_dim
        self.n_classes = config.num_model_number
        self.margin = config.trainer.loss.margin
        self.s = config.trainer.loss.s
        self.W = Parameter(torch.FloatTensor(self.n_classes, self.num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings, labels, index=None):
        x = F.normalize(embeddings)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.margin)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        if index is not None:
            perm_one_hot = torch.zeros_like(one_hot)
            perm_one_hot.scatter_(1, labels[index].view(-1, 1).long(), 1)
            one_hot = one_hot + perm_one_hot
        output = logits * (1 - one_hot) + target_logits * one_hot
        output *= self.s

        return output


class AdaCos(nn.Module):
    def __init__(self, config):
        super(AdaCos, self).__init__()
        self.num_features = config.embedding_dim
        self.n_classes = config.num_model_number
        self.margin = config.trainer.loss.margin
        self.s = math.sqrt(2) * math.log(self.n_classes - 1)
        self.W = Parameter(torch.FloatTensor(self.n_classes, self.num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings, labels, index=None):
        x = F.normalize(embeddings)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.margin)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        if index is not None:
            perm_one_hot = torch.zeros_like(one_hot)
            perm_one_hot.scatter_(1, labels[index].view(-1, 1).long(), 1)
            one_hot = one_hot + perm_one_hot
        output = logits * (1 - one_hot) + target_logits * one_hot
        with torch.no_grad():
            B_avg = torch.where(
                one_hot < 1, self.s * torch.exp(logits), torch.zeros_like(logits)
            )
            B_avg = torch.sum(B_avg) / embeddings.size(0)

            theta_med = torch.median(theta)
            self.s = torch.log(B_avg) / torch.cos(
                torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med)
            )
        output *= self.s

        return output


class CurricularFace(nn.Module):
    def __init__(self, config):
        super(CurricularFace, self).__init__()
        self.in_features = config.embedding_dim
        self.out_features = config.num_model_number
        self.margin = config.trainer.loss.margin
        self.s = config.trainer.loss.s
        self.kernel = Parameter(torch.Tensor(self.in_features, self.out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def l2_norm(x, axis=1):
        norm = torch.norm(x, 2, axis, True)
        output = torch.div(x, norm)
        return output

    def forward(self, embeddings, labels):
        embeddings = self.l2_norm(embeddings, axis=1)
        kernel_norm = self.l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        target_logit = cos_theta[torch.arange(0, embeddings.size(0)), labels].view(
            -1, 1
        )
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * math.cos(self.margin) - sin_theta * math.sin(
            self.margin
        )
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(
            target_logit > math.cos(math.pi - self.margin),
            cos_theta_m,
            target_logit - math.sin(math.pi - self.margin) * self.margin,
        )
        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, labels.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, label):
        logp = self.ce(embeddings, label)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
