import random

import torch
import cv2
import numpy as np
from PIL import Image


def background_removal(PIL_img):
    """背景除去

    Args:
        PIL_img: pillowで読み込んだ画像

    Retrun:
        img: 背景除去された画像
    """
    img = np.array(PIL_img, dtype=np.uint8)
    height, width = img.shape[:2]
    img = cv2.resize(img, dsize=(300, 300))
    edge = cv2.Canny(img, 100, 500)
    kernel = np.ones((5, 5), np.uint8)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(edge.shape, dtype=np.uint8)
    for cont in contours:
        cv2.fillConvexPoly(mask, cont, color=1)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img *= mask
    img = cv2.resize(img, dsize=(width, height))
    img = Image.fromarray(img)
    return img


def mixup_data(x, gpu, mixup_prob=0.5, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if random.uniform(0, 1) < mixup_prob:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda(gpu)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, index, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, gpu, cutmix_prob=0.5, alpha=1.0):
    if random.uniform(0, 1) < cutmix_prob:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = torch.randperm(x.size()[0]).cuda(gpu)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, index, lam


def manifold_mixup(embedding, gpu, mixup_prob=0.5, alpha=1.0):
    if random.uniform(0, 1) < mixup_prob:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = embedding.size()[0]
    index = torch.randperm(batch_size).cuda(gpu)

    mixed_embedding = lam * embedding + (1 - lam) * embedding[index, :]
    return mixed_embedding, index, lam
