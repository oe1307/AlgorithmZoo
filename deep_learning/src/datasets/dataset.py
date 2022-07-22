import re

from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset as _Dataset
from torchvision import transforms as T
from sklearn.preprocessing import LabelEncoder


class Dataset(_Dataset):
    def __init__(self, database, image_path_list, train=False):
        self.image_path_list = image_path_list

        model_numbers = [re.search("[0-9]+", p).group() for p in self.image_path_list]
        self.model_number_encoder = LabelEncoder()
        self.model_numbers = self.model_number_encoder.fit_transform(model_numbers)

        categories = [database.metadata[m]["category"] for m in model_numbers]
        self.categories = LabelEncoder().fit_transform(categories)

        colors = [database.metadata[m]["color"] for m in model_numbers]
        self.colors = LabelEncoder().fit_transform(colors)

        # 型番数、カテゴリ数、色の数を求める
        self.num_model_number = len(set(model_numbers))
        self.num_category = len(set(categories))
        self.num_color = len(set(colors))

        self.transform = T.Compose(
            [T.ToTensor(), T.Normalize(database.mean, database.std)]
        )
        self.argumentation = database.argumentation if train else lambda x: x
        self.resize = T.Resize(
            (database.image_size, database.image_size),
            interpolation=T.InterpolationMode.NEAREST,
        )

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image = Image.open(self.image_path_list[index])
        model_number = self.model_numbers[index]
        category = self.categories[index]
        color = self.colors[index]

        image = self.transform(image)
        image = self.argumentation(image)
        image = self.resize(image)

        return image, model_number, category, color
