import json
from argparse import ArgumentParser

from datetime import datetime
from torch.utils.data import DataLoader

from models import get_model
from datasets import get_dataset
from trainers import get_trainer
from utils import setup_logger, make_config, rename_dir


def argparser():
    parser = ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-tb",
        "--train_batch",
        type=int,
        default=12,
    )
    parser.add_argument(
        "-vb",
        "--valid_batch",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--thread",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        nargs="+",
        help="cuda device, i.e. 0 or 0 1 2 3",
    )
    parser.add_argument(
        "-o",
        "--save_dir",
        default="tmp",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    return parser


def main(config):
    if config.validation:
        # 学習用のデータセットと未知型番データのデータセットを定義
        database = get_dataset(config)

        logger.info(f"train: {len(database.train)}")
        logger.info(f"valid: {len(database.valid)}")
        logger.info(f"reference: {len(database.reference)}")
        logger.info(f"unknown: {len(database.unknown)}")

        train_dataloader = DataLoader(
            database.train,
            shuffle=True,
            drop_last=True,
            batch_size=config.train_batch,
            num_workers=config.thread,
            pin_memory=True,
        )
        valid_dataloader = DataLoader(
            database.valid,
            shuffle=False,
            drop_last=False,
            batch_size=config.valid_batch,
            num_workers=config.thread,
            pin_memory=True,
        )
        embedding_dataloader = DataLoader(
            database.train,
            shuffle=False,
            drop_last=False,
            batch_size=config.valid_batch,
            num_workers=config.thread,
            pin_memory=True,
        )
        reference_dataloader = DataLoader(
            database.reference,
            shuffle=False,
            drop_last=False,
            batch_size=config.valid_batch,
            num_workers=config.thread,
            pin_memory=True,
        )
        unknown_dataloader = DataLoader(
            database.unknown,
            shuffle=False,
            drop_last=False,
            batch_size=config.valid_batch,
            num_workers=config.thread,
            pin_memory=True,
        )
        dataloader = (
            train_dataloader,
            valid_dataloader,
            embedding_dataloader,
            reference_dataloader,
            unknown_dataloader,
        )

    else:
        # 学習用のデータセットを定義
        database = get_dataset(config)

        # 学習用のデータローダーを定義
        train_dataloader = DataLoader(
            database.train,
            shuffle=True,
            drop_last=True,
            batch_size=config.train_batch,
            num_workers=config.thread,
            pin_memory=True,
        )
        dataloader = (train_dataloader, None, None, None, None)

    # モデルを定義
    model = get_model(
        model_name=config.model_name,
        num_model_number=database.train.num_model_number,
        num_category=database.train.num_category,
        num_color=database.train.num_color,
    )

    # trainerを定義する
    config.num_model_number = database.train.num_model_number
    trainer = get_trainer(model, config, dataloader)
    trainer.train()


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    logger = setup_logger.setLevel(args.log_level)
    config = make_config(args.config, args)
    if config.validation:
        config.save_dir = rename_dir("../result/valid/" + config.save_dir)
    else:
        config.save_dir = rename_dir("../result/all_data/" + config.save_dir)
    config.datatime = datetime.now().strftime("%Y%m%d_%H%M%S")
    json.dump(config, open(config.save_dir + "/config.json", "w"), indent=4)
    main(config)
