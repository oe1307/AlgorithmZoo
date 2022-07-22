import timm
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(self, model_size, num_model_number, num_category, num_color):
        super().__init__()
        embedding_dim = {
            "vit_tiny_patch16_224": 192,
            "vit_tiny_patch16_384": 192,
            "vit_small_patch32_224": 384,
            "vit_small_patch32_384": 384,
            "vit_small_patch16_224": 384,
            "vit_small_patch16_384": 384,
            "vit_base_patch32_224": 768,
            "vit_base_patch32_384": 768,
            "vit_base_patch16_224": 768,
            "vit_base_patch16_384": 768,
        }

        self.model = timm.create_model(model_size, pretrained=True, num_classes=0)
        self.embedding_dim = embedding_dim[model_size]
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
