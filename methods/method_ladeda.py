import os
import random
import time
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler, autocast

# ===== IMPORT LaDeDa =====
from LaDeDa import LaDeDa9   # <<< QUAN TRỌNG

# ===================== Utils =====================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def bin_metrics_from_logits(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits).squeeze(1)
    preds = (probs >= threshold).long()
    targets = targets.long()

    TP = ((preds == 1) & (targets == 1)).sum().item()
    TN = ((preds == 0) & (targets == 0)).sum().item()
    FP = ((preds == 1) & (targets == 0)).sum().item()
    FN = ((preds == 0) & (targets == 1)).sum().item()

    eps = 1e-9
    acc = (TP + TN) / max(TP + TN + FP + FN, 1)
    prec = TP / max(TP + FP, 1) if (TP + FP) > 0 else 0.0
    rec = TP / max(TP + FN, 1) if (TP + FN) > 0 else 0.0
    f1 = 2 * prec * rec / max(prec + rec, eps)

    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN}


# ===================== Data =====================

def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, target, path


def build_loaders(data_root, img_size, batch_size, num_workers):
    train_tf, eval_tf = build_transforms(img_size)

    train_ds = CustomImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    val_ds   = CustomImageFolder(os.path.join(data_root, "val"), transform=eval_tf)
    test_ds  = CustomImageFolder(os.path.join(data_root, "test"), transform=eval_tf)

    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers)
    )


# ===================== MODEL =====================

def build_model():
    model = LaDeDa9(
        preprocess_type="NPR",
        num_classes=1,
        pool=True
    )
    return model


# ===================== Train / Eval =====================

def run_one_epoch(loader, model, criterion, optimizer=None, device="cpu", scaler=None):
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum, total = 0.0, 0
    metric_sum = {}
    video_preds = defaultdict(lambda: {"logits": [], "label": -1})

    for imgs, labels, paths in loader:
        video_ids = [os.path.basename(os.path.dirname(p)) for p in paths]
        imgs = imgs.to(device)
        labels = labels.float().to(device)

        if is_train:
            optimizer.zero_grad()
            with autocast(enabled=scaler is not None):
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)

        for i, vid in enumerate(video_ids):
            video_preds[vid]["logits"].append(logits[i].cpu())
            video_preds[vid]["label"] = int(labels[i].cpu())

        batch_metrics = bin_metrics_from_logits(logits.unsqueeze(1), labels)
        bs = imgs.size(0)
        total += bs
        loss_sum += loss.item() * bs

        for k in batch_metrics:
            metric_sum[k] = metric_sum.get(k, 0) + batch_metrics[k] * bs

    out = {k: metric_sum[k] / total for k in ["acc", "precision", "recall", "f1"]}
    out["loss"] = loss_sum / total
    return out


# ===================== MAIN =====================

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader = build_loaders(
        args.data_root, args.img_size, args.batch_size, args.num_workers
    )

    model = build_model().to(device)
    print("Trainable params:", count_params(model))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=device == "cuda")

    best_f1 = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_one_epoch(train_loader, model, criterion, optimizer, device, scaler)
        val_m = run_one_epoch(val_loader, model, criterion, None, device)

        print(f"[{epoch}/{args.epochs}] "
              f"Train F1 {train_m['f1']:.4f} | "
              f"Val F1 {val_m['f1']:.4f} | "
              f"time {time.time()-t0:.1f}s")

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_ladeda.pt"))

    print("Training done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", default="./outputs_ladeda")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
