import timm
import torch
import numpy as np
import torch.nn as nn


class EfficientNet(nn.Module):
    def __init__(self, model_name, num_model_number, num_category, num_color):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            in_chans=3,
            pretrained=True,
            num_classes=0,
            features_only=True,
            out_indices=(3, 4),
        )

        self.global_pools = torch.nn.ModuleList([GeM()] * 2)
        self.embedding_dim = np.sum(self.model.feature_info.channels())
        self.neck = nn.BatchNorm1d(self.embedding_dim)
        # self.neck = nn.LayerNorm(self.embedding_dim)

        self.linear_for_model_number = nn.Linear(self.embedding_dim, num_model_number)
        self.linear_for_category = nn.Linear(self.embedding_dim, num_category)
        self.linear_for_color = nn.Linear(self.embedding_dim, num_color)

        # 線形層の重みを初期化
        nn.init.xavier_uniform_(self.linear_for_category.weight)
        nn.init.xavier_uniform_(self.linear_for_model_number.weight)
        nn.init.xavier_uniform_(self.linear_for_color.weight)

    def forward(self, x):
        ms = self.model(x)
        h = torch.cat([p(m) for m, p in zip(ms, self.global_pools)], dim=1)
        embedding = self.neck(h)
        pred_model_number = self.linear_for_model_number(embedding)
        pred_category = self.linear_for_category(embedding)
        pred_color = self.linear_for_color(embedding)

        return embedding, pred_model_number, pred_category, pred_color


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad=False):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return x.clamp(min=self.eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)
