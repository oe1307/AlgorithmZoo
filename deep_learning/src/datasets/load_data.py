import os
import json
from glob import glob

from attrdict import AttrDict
from torch.utils.data import random_split

from utils import setup_logger

logger = setup_logger(__name__)


def load_data(config):
    database = AttrDict()

    database.metadata = json.load(open(config.dataset.train_metadata))

    if config.validation:
        remove = list(database.metadata.keys())[-config.dataset.num_remove :]

        # データを仕分ける
        train = []
        valid = []
        reference = []
        unknown = []
        for model_number in os.listdir(config.dataset.train):
            if model_number in remove:
                files = glob(os.path.join(config.dataset.train, model_number, "*"))
                valid_size = max(1, int(len(files) * config.dataset.valid))
                reference_size = len(files) - valid_size
                _reference, _unknown = random_split(files, [reference_size, valid_size])
                reference += _reference
                unknown += _unknown
            else:
                files = glob(os.path.join(config.dataset.train, model_number, "*"))
                valid_size = max(1, int(len(files) * config.dataset.valid))
                train_size = len(files) - valid_size
                _train, _valid = random_split(files, [train_size, valid_size])
                train += _train
                valid += _valid
        database.train = train
        database.valid = valid
        database.reference = reference
        database.unknown = unknown

    else:
        database.train = glob(os.path.join(config.dataset.train, "*/*"))

    assert len(database.train) > 0

    if config.debug:
        logger.warning("debug mode")
        database.train = list(database.train)[:24]

    return database
