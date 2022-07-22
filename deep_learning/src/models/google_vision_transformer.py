import torch.nn as nn

from transformers import ViTModel


class GoogleVisionTransformer(nn.Module):
    def __init__(self, model_name, num_model_number, num_category, num_color):
        super().__init__()
        embedding_dim = {
            "google/vit-base-patch16-224": 768,
            "google/vit-base-patch32-384": 768,
            "google/vit-large-patch16-224": 1024,
            "google/vit-large-patch32-384": 1024,
            "google/vit-huge-patch14-224": 1280,
        }

        self.model = ViTModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim[model_name]
        self.linear_for_model_number = nn.Linear(self.embedding_dim, num_model_number)
        self.linear_for_category = nn.Linear(self.embedding_dim, num_category)
        self.linear_for_color = nn.Linear(self.embedding_dim, num_color)

        # 型番、カテゴリ、色を予測する線形層の重みを初期化
        nn.init.xavier_uniform_(self.linear_for_category.weight)
        nn.init.xavier_uniform_(self.linear_for_model_number.weight)
        nn.init.xavier_uniform_(self.linear_for_color.weight)

    def forward(self, x):
        outputs = self.model(x)
        embedding = outputs.last_hidden_state[:, 0, :]

        pred_model_number = self.linear_for_model_number(embedding)
        pred_category = self.linear_for_category(embedding)
        pred_color = self.linear_for_color(embedding)

        return embedding, pred_model_number, pred_category, pred_color
