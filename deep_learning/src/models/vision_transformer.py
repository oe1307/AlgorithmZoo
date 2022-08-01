import timm
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(self, model_name, num_model_number, num_category, num_color):
        super().__init__()
        self.backbone = timm.create_model(
            model_name=model_name, pretrained=True, num_classes=0
        )
        self.embedding_dim = self.backbone.num_features
        self.linear_for_model_number = nn.Linear(self.embedding_dim, num_model_number)
        self.linear_for_category = nn.Linear(self.embedding_dim, num_category)
        self.linear_for_color = nn.Linear(self.embedding_dim, num_color)

    def forward(self, x):
        embedding = self.backbone(x)
        pred_model_number = self.linear_for_model_number(embedding)
        pred_category = self.linear_for_category(embedding)
        pred_color = self.linear_for_color(embedding)

        return embedding, pred_model_number, pred_category, pred_color
