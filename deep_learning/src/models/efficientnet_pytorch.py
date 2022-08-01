import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetPytorch(nn.Module):
    def __init__(self, model_name, num_model_number, num_category, num_color):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(model_name)
        self.embedding_dim = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.linear_for_model_number = nn.Linear(self.embedding_dim, num_model_number)
        self.linear_for_category = nn.Linear(self.embedding_dim, num_category)
        self.linear_for_color = nn.Linear(self.embedding_dim, num_color)

    def forward(self, x):
        embedding = self.backbone(x)
        pred_model_number = self.linear_for_model_number(embedding)
        pred_category = self.linear_for_category(embedding)
        pred_color = self.linear_for_color(embedding)

        return embedding, pred_model_number, pred_category, pred_color
