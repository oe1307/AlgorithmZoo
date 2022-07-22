import timm
import torch.nn as nn


class SwinTransformer(nn.Module):
    def __init__(self, model_name, num_model_number, num_category, num_color):
        super().__init__()
        embedding_dim = {
            "swin_tiny_patch4_window7_224": 768,
            "swin_small_patch4_window7_224": 768,
            "swin_base_patch4_window12_384": 1024,
            "swin_base_patch4_window7_224": 1024,
            "swin_large_patch4_window12_384": 1536,
            "swin_large_patch4_window7_224": 1536,
        }

        self.model = timm.create_model(model_name, pretrained=True)
        self.embedding_dim = embedding_dim[model_name]
        self.linear_for_model_number = nn.Linear(self.embedding_dim, num_model_number)
        self.linear_for_category = nn.Linear(self.embedding_dim, num_category)
        self.linear_for_color = nn.Linear(self.embedding_dim, num_color)

        # 型番、カテゴリ、色を予測する線形層の重みを初期化
        nn.init.xavier_uniform_(self.linear_for_category.weight)
        nn.init.xavier_uniform_(self.linear_for_model_number.weight)
        nn.init.xavier_uniform_(self.linear_for_color.weight)

    def forward(self, x):
        # 注意：最後のclasifierを通してない
        output = self.model.patch_embed(x)
        output = self.model.pos_drop(output)
        output = self.model.layers(output)
        output = self.model.norm(output)
        embedding = output.mean(dim=1)

        pred_model_number = self.linear_for_model_number(embedding)
        pred_category = self.linear_for_category(embedding)
        pred_color = self.linear_for_color(embedding)

        return embedding, pred_model_number, pred_category, pred_color
