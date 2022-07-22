import timm
import torch.nn as nn


class SwinTransformerV2(nn.Module):
    def __init__(self, model_name, num_model_number, num_category, num_color):
        super().__init__()
        embedding_dim = {
            "swinv2_small_window8_256": 768,
            "swinv2_small_window16_256": 768,
            "swinv2_base_window8_256": 1024,
            "swinv2_base_window16_256": 1024,
            "swinv2_large_window12_256": 1536,
            "swinv2_large_window16_256": 1536,
        }

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.embedding_dim = embedding_dim[model_name]
        self.linear_for_model_number = nn.Linear(self.embedding_dim, num_model_number)
        self.linear_for_category = nn.Linear(self.embedding_dim, num_category)
        self.linear_for_color = nn.Linear(self.embedding_dim, num_color)

        # 線形層の重みを初期化
        nn.init.xavier_uniform_(self.linear_for_category.weight)
        nn.init.xavier_uniform_(self.linear_for_model_number.weight)
        nn.init.xavier_uniform_(self.linear_for_color.weight)

    def forward(self, x):
        embedding = self.model(x)
        pred_model_number = self.linear_for_model_number(embedding)
        pred_category = self.linear_for_category(embedding)
        pred_color = self.linear_for_color(embedding)

        return embedding, pred_model_number, pred_category, pred_color
