import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from scipy import spatial
from torchvision import transforms
from torch.nn.parallel import DataParallel

from .loss import FocalLoss, ArcFace, AdaCos, CurricularFace
from .argumentation import mixup_data, cutmix_data, manifold_mixup
from utils import setup_logger

logger = setup_logger(__name__)


class MetricLearningTrainer:
    def __init__(self, model, config, dataloader):
        self.train_dataloader = dataloader[0]
        self.valid_dataloader = dataloader[1]
        self.embedding_dataloader = dataloader[2]
        self.reference_dataloader = dataloader[3]
        self.unknown_dataloader = dataloader[4]

        self.device = min(config.gpu)
        self.device_ids = config.gpu

        self.config = config

        self.to_pillow = transforms.Compose([transforms.ToPILImage()])

        # モデルを並列化
        self.model = self.set_cuda(model)

        # Metric Learning手法を選ぶ
        config.embedding_dim = model.embedding_dim
        if config.trainer.loss.name == "arcface":
            self.margin_penalty = ArcFace(config).to(self.device)
        elif config.trainer.loss.name == "adacos":
            self.margin_penalty = AdaCos(config).to(self.device)
        elif config.trainer.loss.name == "curricularface":
            self.margin_penalty = CurricularFace(config).to(self.device)
        else:
            raise ValueError("Unknown loss function")

        # 損失関数を設定
        if config.trainer.loss.criterion == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif config.trainer.loss.criterion == "focal_loss":
            self.criterion = FocalLoss(gamma=config.loss.gamma_for_focal_loss)
        else:
            raise ValueError("criterion is not defined")

        # 最適化関数を設定
        params = [
            {
                "params": model.model.parameters(),
                "lr": config.trainer.optimizer.lr_pretrained_model,
            },
            {
                "params": model.linear_for_category.parameters(),
                "lr": config.trainer.optimizer.lr_linear_for_category,
            },
            {
                "params": model.linear_for_color.parameters(),
                "lr": config.trainer.optimizer.lr_linear_for_color,
            },
            {
                "params": self.margin_penalty.parameters(),
                "lr": config.trainer.optimizer.lr_margin_penalty,
            },
        ]
        if config.trainer.optimizer.name == "adam":
            self.optimizer = optim.Adam(
                params=params,
                betas=(config.trainer.optimizer.beta1, config.trainer.optimizer.beta2),
                weight_decay=config.trainer.optimizer.weight_decay,
            )
        elif config.trainer.optimizer.name == "adamw":
            self.optimizer = optim.AdamW(
                params=params,
                betas=(config.trainer.optimizer.beta1, config.trainer.optimizer.beta2),
                weight_decay=config.trainer.optimizer.weight_decay,
            )
        elif config.optimizer.name == "sgd":
            self.optimizer = optim.SGD(params=params, weight_decay=config.weight_decay)
        else:
            raise ValueError("optimizer is not defined")

        # マルチタスク学習に関する設定
        if config.trainer.multi_task.model_number.loss == "cross_entropy":
            self.model_number_criterion = nn.CrossEntropyLoss()
        elif config.trainer.multi_task.model_number.loss == "focal_loss":
            self.model_number_criterion = FocalLoss(
                gamma=config.trainer.multi_task.model_number.gamma
            )
        else:
            raise ValueError("model_number_criterion is not defined")

        if config.trainer.multi_task.category.loss == "cross_entropy":
            self.category_criterion = nn.CrossEntropyLoss()
        elif config.trainer.multi_task.category.loss == "focal_loss":
            self.category_criterion = FocalLoss(
                gamma=config.trainer.multi_task.category.gamma
            )
        else:
            raise ValueError("category_criterion is not defined")

        if config.trainer.multi_task.color.loss == "cross_entropy":
            self.color_criterion = nn.CrossEntropyLoss()
        elif config.trainer.multi_task.color.loss == "focal_loss":
            self.color_criterion = FocalLoss(
                gamma=config.trainer.multi_task.color.gamma
            )
        else:
            raise ValueError("color_criterion is not defined")

    # モデルの並列化を行うメソッド
    def set_cuda(self, model):
        if len(self.device_ids) == 1:
            return model.to(self.device)
        else:
            return DataParallel(model.to(self.device), device_ids=self.device_ids)

    def train(self):
        # 高速化
        torch.backends.cudnn.benchmark = True

        if self.config.validation:
            print(
                "epoch,score,valid score,unknown score",
                file=open(self.config.save_dir + "/result.csv", "w"),
            )
            for epoch in range(self.config.trainer.num_epoch):
                # train
                self.model.train()
                self.train_per_epoch(epoch)

                # valid
                if epoch % self.config.dataset.valid_epoch == 0:
                    self.model.eval()
                    embedding_info = self.embedding(epoch)
                    score = self.valid_per_epoch(embedding_info, epoch)
                    print(
                        f"score: {score[0]:.2f} "
                        + f"valid: {score[1]:.2f} unknown: {score[2]:.2f}\n"
                    )
                    print(
                        f"{epoch},{score[0]},{score[1]},{score[2]}",
                        file=open(self.config.save_dir + "/result.csv", "a"),
                    )
        else:
            if len(self.device_ids) > 1:
                for epoch in range(self.config.trainer.num_epoch):
                    # train
                    self.model.train()
                    self.train_per_epoch(epoch)
                    torch.save(
                        self.model.module.state_dict(),
                        self.config.save_dir + f"model_epoch{epoch + 1}.pth",
                    )
            else:
                for epoch in range(self.config.trainer.num_epoch):
                    # train
                    self.model.train()
                    self.train_per_epoch(epoch)
                    torch.save(
                        self.model.state_dict(),
                        f"{self.config.save_dir}/model_epoch{epoch + 1}.pth",
                    )

    # 学習時の1epochの処理メソッド
    @torch.enable_grad()
    def train_per_epoch(self, epoch):
        for i, data in enumerate(self.train_dataloader):
            step = int((i + 1) / len(self.train_dataloader) * 10)
            print(
                f"\rtrain epoch:{epoch + 1} ["
                + "#" * step
                + " " * (10 - step)
                + f"] {i + 1}/{len(self.train_dataloader)}",
                end="",
            )
            # 画像、型番、カテゴリ、色
            images, model_numbers, categories, colors = data
            if self.config.debug:
                self.to_pillow(images[0]).save(self.config.save_dir + f"/{epoch}.png")

            # deviceへ送る
            images = images.to(self.device)
            model_numbers = model_numbers.to(self.device)
            categories = categories.to(self.device)
            colors = colors.to(self.device)

            multi_task = self.config.trainer.multi_task

            # 勾配を初期化
            self.optimizer.zero_grad()

            # モデルへ入力して、Arcface層へ入力する
            if self.config.trainer.argumentation.name == "mixup":
                images, index, lam = mixup_data(
                    images,
                    self.device,
                    mixup_prob=self.config.trainer.argumentation.probability,
                    alpha=self.config.trainer.argumentation.alpha,
                )
                embedding, pred_model_number, pred_category, pred_color = self.model(
                    images
                )
                outputs = self.margin_penalty(embedding, model_numbers, index)
            elif self.config.trainer.argumentation.name == "cutmix":
                images, index, lam = cutmix_data(
                    images,
                    self.device,
                    cutmix_prob=self.config.trainer.argumentation.probability,
                    alpha=self.config.trainer.argumentation.alpha,
                )
                embedding, pred_model_number, pred_category, pred_color = self.model(
                    images
                )
                outputs = self.margin_penalty(embedding, model_numbers, index)
            elif self.config.trainer.argumentation.name == "manifold_mixUp":
                embedding, pred_model_number, pred_category, pred_color = self.model(
                    images
                )
                embedding, index, lam = manifold_mixup(
                    embedding,
                    self.device,
                    mixup_prob=self.config.trainer.argumentation.probability,
                    alpha=self.config.trainer.argumentation.alpha,
                )
                outputs = self.margin_penalty(embedding, model_numbers, index)
            elif self.config.trainer.argumentation.name is None:
                embedding, pred_model_number, pred_category, pred_color = self.model(
                    images
                )
                index = torch.randperm(images.shape[0]).cuda(self.device)
                lam = 1
                outputs = self.margin_penalty(embedding, model_numbers)
            else:
                raise ValueError("argumentation_type is not supported.")

            loss = lam * self.criterion(outputs, model_numbers)
            loss += (1 - lam) * self.criterion(outputs, model_numbers[index])

            if multi_task.model_number.flag:
                loss += (
                    multi_task.model_number.weight
                    * lam
                    * self.model_number_criterion(pred_model_number, model_numbers)
                )
                loss += (
                    multi_task.model_number.weight
                    * (1 - lam)
                    * self.model_number_criterion(
                        pred_model_number, model_numbers[index]
                    )
                )
            if multi_task.category.flag:
                loss += (
                    multi_task.category.weight
                    * lam
                    * self.category_criterion(pred_category, categories)
                )
                loss += (
                    multi_task.category.weight
                    * (1 - lam)
                    * self.category_criterion(pred_category, categories[index])
                )
            if multi_task.color.flag:
                loss += (
                    multi_task.color.weight
                    * lam
                    * self.color_criterion(pred_color, colors)
                )
                loss += (
                    multi_task.color.weight
                    * (1 - lam)
                    * self.color_criterion(pred_color, colors[index])
                )

            # optimizerを逆伝搬計算、勾配クリッピング、パラメータを更新
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.trainer.max_norm
            )
            self.optimizer.step()

            del loss

        print(f"\rtrain epoch:{epoch + 1} [" + "#" * 10 + f"] {i + 1}/{i + 1}")

    # Embedding空間に埋め込むメソッド
    @torch.no_grad()
    def embedding(self, epoch):
        # 埋め込みベクトル情報を格納する辞書を定義
        embedding_info = []
        model_number_encoder = self.embedding_dataloader.dataset.model_number_encoder

        for i, data in enumerate(self.embedding_dataloader):
            step = int((i + 1) / len(self.embedding_dataloader) * 10)
            print(
                f"\rembedding epoch:{epoch + 1} ["
                + "#" * step
                + " " * (10 - step)
                + f"] {i + 1}/{len(self.embedding_dataloader)}",
                end="",
            )
            # 画像、型番、カテゴリ、色
            images, model_numbers = data[:2]
            images = images.to(self.device)
            model_numbers = model_number_encoder.inverse_transform(model_numbers)

            embedding = self.model(images)[0]
            embedding = embedding.detach().to("cpu").numpy()

            # 埋め込みベクトル情報を追加していく
            for embed, model_number in zip(embedding, model_numbers):
                embedding_info.append((model_number, embed))

        print(f"\rembedding epoch:{epoch + 1} [" + "#" * 10 + f"] {i + 1}/{i + 1}")

        model_number_encoder = self.reference_dataloader.dataset.model_number_encoder
        for i, data in enumerate(self.reference_dataloader):
            step = int((i + 1) / len(self.reference_dataloader) * 10)
            print(
                f"\rreference epoch:{epoch + 1} ["
                + "#" * step
                + " " * (10 - step)
                + f"] {i + 1}/{len(self.reference_dataloader)}",
                end="",
            )

            images, model_numbers = data[:2]
            images = images.to(self.device)
            model_numbers = model_number_encoder.inverse_transform(model_numbers)

            embedding = self.model(images)[0]
            embedding = embedding.detach().to("cpu").numpy()

            # 埋め込みベクトル情報を追加していく
            for embed, model_number in zip(embedding, model_numbers):
                embedding_info.append((model_number, embed))

        print(f"\rreference epoch:{epoch + 1} [" + "#" * 10 + f"] {i + 1}/{i + 1}")

        return embedding_info

    @torch.no_grad()
    def valid_per_epoch(self, embedding_info, epoch):
        # 既知型番データローダーのループ処理
        score = [0, 0, 0]
        valid_img_num = 0

        model_number_encoder = self.valid_dataloader.dataset.model_number_encoder
        for i, data in enumerate(self.valid_dataloader):
            print(
                f"\rvalidation epoch:{epoch + 1} ["
                + "#" * int((i + 1) / len(self.valid_dataloader) * 10)
                + " " * int((1 - (i + 1) / len(self.valid_dataloader)) * 10)
                + f"] {i + 1}/{len(self.valid_dataloader)}",
                end="",
            )
            # 画像、型番
            images, model_numbers = data[:2]
            images = images.to(self.device)
            model_numbers = model_number_encoder.inverse_transform(model_numbers)

            # モデルへ入力する
            embedding, pred_model_number = self.model(images)[:2]
            embedding = embedding.detach().to("cpu").numpy()

            # knnに基づく予測を計算する
            if self.config.trainer.knn.method == "center":
                embedding_df = pd.DataFrame(
                    embedding_info, columns=["model_number", "embedding"]
                )
                embedding_df = embedding_df.groupby("model_number").mean()
                distances = spatial.distance.cdist(
                    embedding,
                    list(embedding_df["embedding"].values),
                    self.config.trainer.knn.metric,
                )
                for dist, model_number in zip(distances, model_numbers):
                    valid_img_num += 1
                    embedding_df["distance"] = dist
                    _embedding_df = embedding_df.sort_values("distance")
                    MAPatR = _embedding_df[:10]["model_number"].values.tolist()
                    if model_number in MAPatR:
                        logger.info(f"MAP@10: {MAPatR.index(model_number) + 1}")
                        score[1] += 1 / (MAPatR.index(model_number) + 1)
                    else:
                        logger.info("MAP@10: x")
            elif self.config.trainer.knn.method == "weighted":
                embedding_df = pd.DataFrame(
                    embedding_info, columns=["model_number", "embedding"]
                )
                distances = spatial.distance.cdist(
                    embedding,
                    list(embedding_df["embedding"].values),
                    self.config.trainer.knn.metric,
                )
                for dist, model_number in zip(distances, model_numbers):
                    valid_img_num += 1
                    embedding_df["distance"] = dist
                    _embedding_df = embedding_df.sort_values("distance")
                    _embedding_df["score"] = 1 / np.arange(1, len(embedding_df) + 1)
                    _embedding_df["score"] = _embedding_df.groupby("model_number")[
                        "score"
                    ].sum()
                    _embedding_df = _embedding_df.sort_values("score")
                    MAPatR = _embedding_df[:10]["model_number"].values.tolist()
                    if model_number in MAPatR:
                        logger.info(f"MAP@10: {MAPatR.index(model_number) + 1}")
                        score[1] += 1 / (MAPatR.index(model_number) + 1)
                    else:
                        logger.info("MAP@10: x")
            elif self.config.trainer.knn.method == "nearest":
                embedding_df = pd.DataFrame(
                    embedding_info, columns=["model_number", "embedding"]
                )
                distances = spatial.distance.cdist(
                    embedding,
                    list(embedding_df["embedding"].values),
                    self.config.trainer.knn.metric,
                )
                for dist, model_number in zip(distances, model_numbers):
                    valid_img_num += 1
                    embedding_df["distance"] = dist
                    _embedding_df = embedding_df.sort_values("distance")
                    _embedding_df = _embedding_df.drop_duplicates("model_number")
                    MAPatR = _embedding_df[:10]["model_number"].values.tolist()
                    if model_number in MAPatR:
                        logger.info(f"MAP@10: {MAPatR.index(model_number) + 1}")
                        score[1] += 1 / (MAPatR.index(model_number) + 1)
                    else:
                        logger.info("MAP@10: x")
            elif self.config.trainer.knn.method is None:
                pred_model_number = pred_model_number.detach().to("cpu").numpy()
                MAPatR = np.argsort(pred_model_number)[::-1][:10].tolist()
                for model_number in model_numbers:
                    valid_img_num += 1
                    if model_number in MAPatR:
                        logger.info(f"MAP@10: {MAPatR.index(model_number) + 1}")
                        score[1] += 1 / (MAPatR.index(model_number) + 1)
                    else:
                        logger.info("MAP@10: x")
            else:
                raise ValueError(
                    f"knn.method is invalid. {self.config.trainer.knn.method}"
                )

        print(f"\rvalidation epoch:{epoch + 1} [" + "#" * 10 + f"] {i + 1}/{i + 1}")

        unknown_img_num = 0

        model_number_encoder = self.unknown_dataloader.dataset.model_number_encoder
        for i, data in enumerate(self.unknown_dataloader):
            print(
                f"\runknown epoch:{epoch + 1} ["
                + "#" * int((i + 1) / len(self.unknown_dataloader) * 10)
                + " " * int((1 - (i + 1) / len(self.unknown_dataloader)) * 10)
                + f"] {i + 1}/{len(self.unknown_dataloader)}",
                end="",
            )
            # 画像、型番
            images, model_numbers = data[:2]
            images = images.to(self.device)
            model_numbers = model_number_encoder.inverse_transform(model_numbers)

            # モデルへ入力する
            embedding, pred_model_number = self.model(images)[:2]
            embedding = embedding.to("cpu").numpy()

            if self.config.trainer.knn.method == "center":
                embedding_df = pd.DataFrame(
                    embedding_info, columns=["model_number", "embedding"]
                )
                embedding_df = embedding_df.groupby("model_number").mean()
                distances = spatial.distance.cdist(
                    embedding,
                    list(embedding_df["embedding"].values),
                    self.config.trainer.knn.metric,
                )
                for dist, model_number in zip(distances, model_numbers):
                    unknown_img_num += 1
                    embedding_df["distance"] = dist
                    _embedding_df = embedding_df.sort_values("distance")
                    MAPatR = _embedding_df[:10]["model_number"].values.tolist()
                    if model_number in MAPatR:
                        logger.info(f"MAP@10: {MAPatR.index(model_number) + 1}")
                        score[2] += 1 / (MAPatR.index(model_number) + 1)
                    else:
                        logger.info("MAP@10: x")
            elif self.config.trainer.knn.method == "weighted":
                embedding_df = pd.DataFrame(
                    embedding_info, columns=["model_number", "embedding"]
                )
                distances = spatial.distance.cdist(
                    embedding,
                    list(embedding_df["embedding"].values),
                    self.config.trainer.knn.metric,
                )
                for dist, model_number in zip(distances, model_numbers):
                    unknown_img_num += 1
                    embedding_df["distance"] = dist
                    _embedding_df = embedding_df.sort_values("distance")
                    _embedding_df["score"] = 1 / np.arange(1, len(embedding_df) + 1)
                    _embedding_df["score"] = _embedding_df.groupby("model_number")[
                        "score"
                    ].sum()
                    _embedding_df = _embedding_df.sort_values("score")
                    MAPatR = _embedding_df[:10]["model_number"].values.tolist()
                    if model_number in MAPatR:
                        logger.info(f"MAP@10: {MAPatR.index(model_number) + 1}")
                        score[2] += 1 / (MAPatR.index(model_number) + 1)
                    else:
                        logger.info("MAP@10: x")
            elif self.config.trainer.knn.method == "nearest":
                embedding_df = pd.DataFrame(
                    embedding_info, columns=["model_number", "embedding"]
                )
                distances = spatial.distance.cdist(
                    embedding,
                    list(embedding_df["embedding"].values),
                    self.config.trainer.knn.metric,
                )
                for dist, model_number in zip(distances, model_numbers):
                    unknown_img_num += 1
                    embedding_df["distance"] = dist
                    _embedding_df = embedding_df.sort_values("distance")
                    _embedding_df = _embedding_df.drop_duplicates("model_number")
                    MAPatR = _embedding_df[:10]["model_number"].values.tolist()
                    if model_number in MAPatR:
                        logger.info(f"MAP@10: {MAPatR.index(model_number) + 1}")
                        score[2] += 1 / (MAPatR.index(model_number) + 1)
                    else:
                        logger.info("MAP@10: x")
            elif self.config.trainer.knn.method is None:
                pred_model_number = pred_model_number.detach().to("cpu").numpy()
                MAPatR = np.argsort(pred_model_number)[::-1][:10].tolist()
                for model_number in model_numbers:
                    unknown_img_num += 1
                    if model_number in MAPatR:
                        logger.info(f"MAP@10: {MAPatR.index(model_number) + 1}")
                        score[2] += 1 / (MAPatR.index(model_number) + 1)
                    else:
                        logger.info("MAP@10: x")
            else:
                raise ValueError(
                    f"knn.method is invalid. {self.config.trainer.knn.method}"
                )

        print(f"\runknown epoch:{epoch + 1} [" + "#" * 10 + f"] {i + 1}/{i + 1}")

        score[0] = (score[1] + score[2]) / (valid_img_num + unknown_img_num) * 100
        score[1] = score[1] / valid_img_num * 100
        score[2] = score[2] / unknown_img_num * 100

        return score
