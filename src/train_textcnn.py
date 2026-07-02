# -*- coding: utf-8 -*-
"""
基于 PyTorch Embedding-MLP 的新闻文本 14 分类实验

功能：
1. 读取预处理后的 token 序列
2. 使用训练集内部验证集选择最优模型
3. 在固定 20% 测试集上计算 Accuracy、Macro-F1、分类报告和混淆矩阵
4. 保存模型、训练曲线和评价结果
"""

from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SEED = 42
BATCH_SIZE = 1024
EPOCHS = 12
MODEL_MAX_LEN = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 3
VAL_SIZE = 0.1
CLASS_WEIGHT_POWER = 0.5
EMBED_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.35

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
CHECKPOINT_DIR = ROOT / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NewsDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, labels: np.ndarray):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.input_ids[index], self.labels[index]


class TextCNN(nn.Module):
    """保留 TextCNN 类名以兼容推理脚本，内部使用更适合 CPU 的池化文本分类器。"""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0)
        x = self.embedding(input_ids)

        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pool = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths

        masked_x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_pool = masked_x.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        x = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(x)


def build_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    class_counts = np.bincount(labels, minlength=num_classes)
    raw_weights = len(labels) / (num_classes * class_counts)
    smoothed_weights = np.power(raw_weights, CLASS_WEIGHT_POWER)
    smoothed_weights = smoothed_weights / smoothed_weights.mean()
    return torch.tensor(smoothed_weights, dtype=torch.float32).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for input_ids, labels in tqdm(loader, desc="Training", leave=False):
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes: int):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for input_ids, labels in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * labels.size(0)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    one_hot_labels = np.eye(num_classes)[all_labels]
    abs_sum = np.abs(one_hot_labels - all_probs).sum()

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_precision": precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "abs_sum": abs_sum,
        "mean_abs_sum": abs_sum / len(all_labels),
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
    }


def draw_training_curves(history: dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("News Text Classifier Training: Train vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "train_val_loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_f1"], label="Train Macro-F1")
    plt.plot(epochs, history["val_f1"], label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("News Text Classifier Training: Train vs Validation Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "train_val_f1_curve.png", dpi=300)
    plt.close()


def draw_confusion_matrix(cm: np.ndarray, num_classes: int) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm / np.maximum(row_sums, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("News Text Classifier Training: Final Evaluation on Held-out Test Set")

    raw_image = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Test Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_xticks(range(num_classes))
    axes[0].set_yticks(range(num_classes))
    fig.colorbar(raw_image, ax=axes[0], fraction=0.046, pad=0.04)

    norm_image = axes[1].imshow(cm_normalized, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Test Row Normalized")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_xticks(range(num_classes))
    axes[1].set_yticks(range(num_classes))
    fig.colorbar(norm_image, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIGURE_DIR / "test_evaluation_confusion_matrix.png", dpi=300)
    plt.close()


def main() -> None:
    set_seed(SEED)

    metadata_path = PROCESSED_DIR / "metadata.json"
    split_path = OUTPUT_DIR / "split_indices.npz"
    if not metadata_path.exists():
        raise FileNotFoundError("请先运行：python src/prepare_data.py")
    if not split_path.exists():
        raise FileNotFoundError("请先运行：python src/split_data.py")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vocab_size = int(metadata["vocab_size"])
    num_classes = int(metadata["num_classes"])

    print("========== 实验环境 ==========")
    print(f"设备：{DEVICE}")
    print(f"PyTorch 版本：{torch.__version__}")
    print(f"词表大小：{vocab_size}")
    print(f"类别数：{num_classes}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    input_ids = np.load(PROCESSED_DIR / "input_ids.npy", mmap_mode="r")
    labels = np.load(PROCESSED_DIR / "labels.npy")
    split_data = np.load(split_path)
    train_idx = split_data["train_idx"]
    test_idx = split_data["test_idx"]

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=labels[train_idx],
    )

    x_train = np.asarray(input_ids[train_idx, :MODEL_MAX_LEN])
    y_train = labels[train_idx]
    x_val = np.asarray(input_ids[val_idx, :MODEL_MAX_LEN])
    y_val = labels[val_idx]
    x_test = np.asarray(input_ids[test_idx, :MODEL_MAX_LEN])
    y_test = labels[test_idx]

    print("\n========== 数据划分 ==========")
    print(f"训练集规模：{len(y_train)}")
    print(f"验证集规模：{len(y_val)}")
    print(f"测试集规模：{len(y_test)}")

    train_loader = DataLoader(
        NewsDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        NewsDataset(x_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        NewsDataset(x_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    class_weights = build_class_weights(y_train, num_classes)
    print("\n损失函数类别权重（已平滑）：")
    for label, weight in enumerate(class_weights.cpu().numpy()):
        print(f"label={label}: weight={weight:.4f}")

    model = TextCNN(vocab_size=vocab_size, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_f1 = -1.0
    epochs_without_improvement = 0
    best_model_path = CHECKPOINT_DIR / "best_text_model.pt"

    print("\n========== 开始训练 ==========")
    for epoch in range(1, EPOCHS + 1):
        train_result = train_one_epoch(model, train_loader, criterion, optimizer)
        val_result = evaluate(model, val_loader, criterion, num_classes)
        scheduler.step(val_result["macro_f1"])

        history["train_loss"].append(train_result["loss"])
        history["val_loss"].append(val_result["loss"])
        history["train_f1"].append(train_result["macro_f1"])
        history["val_f1"].append(val_result["macro_f1"])

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_result['loss']:.4f} | "
            f"Train Acc: {train_result['accuracy']:.4f} | "
            f"Train F1: {train_result['macro_f1']:.4f} | "
            f"Val Loss: {val_result['loss']:.4f} | "
            f"Val Acc: {val_result['accuracy']:.4f} | "
            f"Val F1: {val_result['macro_f1']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_result["macro_f1"] > best_f1:
            best_f1 = val_result["macro_f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  保存当前最优模型：Validation Macro-F1={best_f1:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("触发 Early Stopping，停止训练。")
            break

    draw_training_curves(history)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    final_result = evaluate(model, test_loader, criterion, num_classes)
    report = classification_report(
        final_result["labels"],
        final_result["preds"],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(
        final_result["labels"],
        final_result["preds"],
        labels=list(range(num_classes)),
    )
    draw_confusion_matrix(cm, num_classes)

    metrics_text = (
        "训练设置\n"
        f"Epochs: {len(history['train_loss'])}/{EPOCHS}\n"
        f"Best Validation Macro F1: {best_f1:.4f}\n"
        f"Max Len: {metadata['max_len']}\n"
        f"Model Input Len: {MODEL_MAX_LEN}\n"
        f"Embedding Dim: {EMBED_DIM}\n"
        f"Hidden Dim: {HIDDEN_DIM}\n"
        "Model: Embedding mean/max pooling + MLP\n\n"
        "最终测试结果\n"
        f"Accuracy: {final_result['accuracy']:.4f}\n"
        f"Macro Precision: {final_result['macro_precision']:.4f}\n"
        f"Macro Recall: {final_result['macro_recall']:.4f}\n"
        f"Macro F1: {final_result['macro_f1']:.4f}\n"
        f"ABS-SUM: {final_result['abs_sum']:.4f}\n"
        f"Mean ABS-SUM: {final_result['mean_abs_sum']:.6f}\n\n"
        "分类报告\n"
        f"{report}\n\n"
        "混淆矩阵\n"
        f"{cm}\n"
    )
    (OUTPUT_DIR / "metrics.txt").write_text(metrics_text, encoding="utf-8")

    print("\n========== 最终测试结果 ==========")
    print(metrics_text)
    print("========== 输出文件 ==========")
    print(f"最佳模型：{best_model_path}")
    print(f"评价结果：{OUTPUT_DIR / 'metrics.txt'}")
    print(f"训练曲线：{FIGURE_DIR / 'train_val_loss_curve.png'}")
    print(f"F1 曲线：{FIGURE_DIR / 'train_val_f1_curve.png'}")
    print(f"混淆矩阵：{FIGURE_DIR / 'test_evaluation_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
