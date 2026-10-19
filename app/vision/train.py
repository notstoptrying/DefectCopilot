"""训练钢材表面缺陷多类别分割模型（UNet，4 类）。

数据集：NEU-Seg（路径由环境变量 NEU_SEG_DIR 或 data/NEU_Seg 指定）
  images/training/*.jpg      3630 张 200×200 RGB
  annotations/training/*.png 3630 张掩码，取值 0/1/2/3
  images/test + annotations/test  840 张
  数据集未提供类别名称，按掩码形态目视为：
    1 条带状夹杂（area≈5%）
    2 斑块状缺陷（area≈18%）
    3 线状划痕（area≈7%）

划分：官方 train 内部再切 90/10 作训练/验证；官方 test 作为最终测试集。
损失：CrossEntropy + 多类 Dice。
指标：逐类 IoU、mIoU（4 类）、缺陷类平均 IoU、像素精度。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from ..config import DATASET_DIR

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .unet import UNet, count_params

DEFAULT_DATA = DATASET_DIR
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "weights"

# 类别语义（按掩码形态判定，数据集未提供官方名称）
CLASS_NAMES = {0: "背景", 1: "条带状夹杂", 2: "斑块状缺陷", 3: "线状划痕"}
NUM_CLASSES = 4


class NeuSegDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], train: bool = False):
        self.pairs = pairs
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        img_path, mask_path = self.pairs[i]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        x = np.asarray(img, dtype=np.float32) / 255.0
        y = np.asarray(mask, dtype=np.int64)
        if y.shape != x.shape[:2]:
            y = np.asarray(
                Image.fromarray(np.asarray(mask)).resize((x.shape[1], x.shape[0]), Image.NEAREST),
                dtype=np.int64,
            )

        if self.train:
            if random.random() < 0.5:  # 水平翻转
                x, y = np.ascontiguousarray(x[:, ::-1]), np.ascontiguousarray(y[:, ::-1])
            if random.random() < 0.5:  # 垂直翻转（线状缺陷方向多样，翻转有效）
                x, y = np.ascontiguousarray(x[::-1]), np.ascontiguousarray(y[::-1])
            if random.random() < 0.3:  # 亮度扰动
                x = np.clip(x * random.uniform(0.85, 1.15), 0, 1)

        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
        return x, torch.from_numpy(np.ascontiguousarray(y))


def build_splits(data_dir: Path, seed: int = 42):
    def collect(img_dir: Path, ann_dir: Path) -> list[tuple[Path, Path]]:
        return [
            (p, ann_dir / f"{p.stem}.png")
            for p in sorted(img_dir.glob("*.jpg"))
            if (ann_dir / f"{p.stem}.png").exists()
        ]

    train_all = collect(data_dir / "images" / "training", data_dir / "annotations" / "training")
    test = collect(data_dir / "images" / "test", data_dir / "annotations" / "test")
    rng = random.Random(seed)
    rng.shuffle(train_all)
    n_val = int(len(train_all) * 0.1)
    return train_all[:-n_val], train_all[-n_val:], test


def multiclass_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = F.softmax(logits, dim=1)
    target_1h = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    num = 2 * (prob * target_1h).sum(dim=(2, 3)) + eps
    den = prob.sum(dim=(2, 3)) + target_1h.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    inter = np.zeros(NUM_CLASSES, dtype=np.int64)
    union = np.zeros(NUM_CLASSES, dtype=np.int64)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum())
        total += y.numel()
        for c in range(NUM_CLASSES):
            p, t = pred == c, y == c
            inter[c] += int((p & t).sum())
            union[c] += int((p | t).sum())
    ious = inter / np.maximum(union, 1)
    return {
        "per_class_iou": {CLASS_NAMES[c]: round(float(ious[c]), 4) for c in range(NUM_CLASSES)},
        "miou": round(float(ious.mean()), 4),
        "defect_miou": round(float(ious[1:].mean()), 4),
        "pixel_acc": round(correct / max(1, total), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_p, val_p, test_p = build_splits(Path(args.data), args.seed)
    print(f"device={device} train={len(train_p)} val={len(val_p)} test={len(test_p)}", flush=True)

    train_loader = DataLoader(NeuSegDataset(train_p, True), batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(NeuSegDataset(val_p), batch_size=args.batch, num_workers=0)
    test_loader = DataLoader(NeuSegDataset(test_p), batch_size=args.batch, num_workers=0)

    model = UNet(in_ch=3, out_ch=NUM_CLASSES).to(device)
    print(f"params={count_params(model):,}", flush=True)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_miou, best_epoch, history = -1.0, 0, []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = ce(logits, y) + multiclass_dice(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss) * x.size(0)
        sched.step()

        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "loss": round(running / max(1, len(train_loader.dataset)), 4), **metrics})
        if metrics["miou"] > best_miou:
            best_miou, best_epoch = metrics["miou"], ep
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": ep,
                    "val": metrics,
                    "arch": "unet-base32",
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                },
                OUT_DIR / "unet_steel.pt",
            )
        if ep % 5 == 0 or ep == 1:
            print(
                f"ep {ep:3d} loss {history[-1]['loss']:.4f} val mIoU {metrics['miou']:.4f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    ckpt = torch.load(OUT_DIR / "unet_steel.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    report = {
        "dataset": str(args.data),
        "split": {"train": len(train_p), "val": len(val_p), "test": len(test_p)},
        "input_size": 200,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "best_epoch": best_epoch,
        "val_best": ckpt["val"],
        "test": test_metrics,
        "train_seconds": round(time.time() - t0, 1),
        "params": count_params(model),
        "history": history,
    }
    (OUT_DIR / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TEST", json.dumps(test_metrics, ensure_ascii=False), flush=True)
    print(f"done in {report['train_seconds']}s, best_epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
