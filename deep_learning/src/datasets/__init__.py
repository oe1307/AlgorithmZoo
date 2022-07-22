from torchvision import transforms as T

from .load_data import load_data
from .image_size import model2size
from .dataset import Dataset


def get_dataset(config):

    # 画像の読み込み
    database = load_data(config)

    # train時のargumentation
    argumentation = {
        "h_flip": T.RandomHorizontalFlip(p=0.5),
        "v_flip": T.RandomVerticalFlip(p=0.5),
        "rotation": T.RandomRotation(180, expand=True),
        "erasing": T.RandomErasing(scale=(0.01, 0.1), p=0.25),
        "crop": T.RandomResizedCrop(model2size[config.model_name]),
        "blur": T.GaussianBlur(kernel_size=3),
    }
    database.argumentation = T.Compose(
        [argumentation[k] for k in config.dataset.transforms]
    )

    # 画像サイズを決定
    database.image_size = model2size[config.model_name]

    # normalize時のmean, std
    if "google/vit" in config.model_name:
        database.mean = [0.5, 0.5, 0.5]
        database.std = [0.5, 0.5, 0.5]
    elif "swin" or "efficientnet" in config.model_name:
        database.mean = [0.485, 0.456, 0.406]
        database.std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown model name: {config.model_name}")

    if config.validation:
        database.train = Dataset(database, database.train, train=True)
        database.valid = Dataset(database, database.valid)
        database.reference = Dataset(database, database.reference)
        database.unknown = Dataset(database, database.unknown)
    else:
        database.train = Dataset(database, database.train, train=True)

    return database
