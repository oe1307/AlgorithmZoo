import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetPytorch(nn.Module):
    def __init__(self, model_name, num_model_number, num_category, num_color):
        super().__init__()
        embedding_dim = {
            "efficientnet-b0": 1280,
            "efficientnet-b1": 1280,
            "efficientnet-b2": 1408,
            "efficientnet-b3": 1536,
            "efficientnet-b4": 1792,
            "efficientnet-b5": 2048,
            "efficientnet-b6": 2304,
            "efficientnet-b7": 2560,
        }
        self.model = EfficientNet.from_pretrained(model_name)
        self.embedding_dim = embedding_dim[model_name]
        self.linear_for_model_number = nn.Linear(self.embedding_dim, num_model_number)
        self.linear_for_category = nn.Linear(self.embedding_dim, num_category)
        self.linear_for_color = nn.Linear(self.embedding_dim, num_color)

        # 型番、カテゴリ、色を予測する線形層の重みを初期化
        nn.init.xavier_uniform_(self.linear_for_category.weight)
        nn.init.xavier_uniform_(self.linear_for_model_number.weight)
        nn.init.xavier_uniform_(self.linear_for_color.weight)

    def forward(self, x):
        # 注意：最後のclassifierを通してない
        output = self.model._conv_stem(x)
        output = self.model._bn0(output)
        for i in range(len(self.model._blocks)):
            output = self.model._blocks[i](output)
        output = self.model._conv_head(output)
        output = self.model._avg_pooling(output)
        embedding = output.view(output.size(0), output.size(1))

        pred_model_number = self.linear_for_model_number(embedding)
        pred_category = self.linear_for_category(embedding)
        pred_color = self.linear_for_color(embedding)

        return embedding, pred_model_number, pred_category, pred_color
