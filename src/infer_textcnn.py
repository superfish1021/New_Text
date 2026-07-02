# -*- coding: utf-8 -*-
"""
新闻文本分类推理脚本

1. 读取固定 20% 测试集
2. 加载 best_text_model.pt 推理
3. 输出测试集整理文件、预测明细、错分样本和结果分析
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
INFER_DIR = OUTPUT_DIR / "infer"
CHECKPOINT_DIR = ROOT / "checkpoints"

sys.path.insert(0, str(SRC_DIR))
from train_textcnn import MODEL_MAX_LEN, NewsDataset, TextCNN  # noqa: E402


def parse_args():
    parser = ArgumentParser(description="Run text classifier inference on fixed test split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "best_text_model.pt",
        help="Path to trained text classifier checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INFER_DIR,
        help="Directory for inference outputs.",
    )
    return parser.parse_args()


def load_test_split():
    required_paths = [
        PROCESSED_DIR / "input_ids.npy",
        PROCESSED_DIR / "labels.npy",
        PROCESSED_DIR / "ids.npy",
        PROCESSED_DIR / "metadata.json",
        OUTPUT_DIR / "split_indices.npz",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"缺少必要文件：{path}")

    input_ids = np.load(PROCESSED_DIR / "input_ids.npy", mmap_mode="r")
    labels = np.load(PROCESSED_DIR / "labels.npy")
    ids = np.load(PROCESSED_DIR / "ids.npy")
    metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text(encoding="utf-8"))
    split_data = np.load(OUTPUT_DIR / "split_indices.npz")
    test_idx = split_data["test_idx"]

    return (
        np.asarray(input_ids[test_idx, :MODEL_MAX_LEN]),
        labels[test_idx].astype(np.int64),
        ids[test_idx],
        test_idx,
        metadata,
    )


def save_test_set(output_dir: Path, input_ids, labels, ids, test_idx, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "test_input_ids.npy", input_ids)
    np.save(output_dir / "test_labels.npy", labels)
    np.save(output_dir / "test_ids.npy", ids)
    np.save(output_dir / "test_indices.npy", test_idx)

    if not (output_dir / "test_set.csv").exists():
        token_offset = int(metadata.get("token_offset", 1))
        text_values = [
            " ".join(str(int(token) - token_offset) for token in row if int(token) != 0)
            for row in tqdm(input_ids, desc="Writing test_set.csv", leave=False)
        ]
        test_df = pd.DataFrame({
            "id": ids,
            "label": labels,
            "text": text_values,
        })
        test_df.to_csv(output_dir / "test_set.csv", index=False, encoding="utf-8-sig")


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []

    for input_ids, _ in tqdm(loader, desc="Inferencing", leave=False):
        input_ids = input_ids.to(device)
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    confidence = probs.max(axis=1)
    return preds, probs, confidence


def save_prediction_files(output_dir, ids, labels, preds, probs, confidence, num_classes):
    prob_columns = [f"label_{i}" for i in range(num_classes)]

    submit_df = pd.DataFrame(probs, columns=prob_columns)
    submit_df.insert(0, "id", ids)
    submit_df.to_csv(output_dir / "submit_result.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "id": ids,
        "true_label": labels,
        "pred_label": preds,
        "confidence": confidence,
    })
    for i, column in enumerate(prob_columns):
        pred_df[column] = probs[:, i]
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]
    pred_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    wrong_df = pred_df[~pred_df["correct"]].copy()
    wrong_df = wrong_df.sort_values(
        by=["true_label", "pred_label", "confidence"],
        ascending=[True, True, False],
    )
    wrong_df.to_csv(output_dir / "misclassified.csv", index=False, encoding="utf-8-sig")

    np.save(output_dir / "pred_probs.npy", probs)
    np.save(output_dir / "pred_labels.npy", preds)
    return pred_df, wrong_df


def plot_confusion_matrix(output_dir: Path, cm: np.ndarray, num_classes: int):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm / np.maximum(row_sums, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("News Text Classifier Inference: Held-out 20% Test Set")

    raw_image = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Inference Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_xticks(range(num_classes))
    axes[0].set_yticks(range(num_classes))
    fig.colorbar(raw_image, ax=axes[0], fraction=0.046, pad=0.04)

    norm_image = axes[1].imshow(cm_normalized, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Inference Row Normalized")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_xticks(range(num_classes))
    axes[1].set_yticks(range(num_classes))
    fig.colorbar(norm_image, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "inference_confusion_matrix.png", dpi=300)
    plt.close()


def build_analysis(labels, preds, probs, confidence, cm, checkpoint, metadata):
    num_classes = int(metadata["num_classes"])
    accuracy = accuracy_score(labels, preds)
    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    report = classification_report(labels, preds, digits=4, zero_division=0)

    one_hot_labels = np.eye(num_classes)[labels]
    abs_sum = np.abs(one_hot_labels - probs).sum()
    mean_abs_sum = abs_sum / len(labels)

    true_unique, true_counts = np.unique(labels, return_counts=True)
    pred_unique, pred_counts = np.unique(preds, return_counts=True)
    true_distribution = {
        int(label): int(count)
        for label, count in zip(true_unique, true_counts)
    }
    pred_distribution = {
        int(label): int(count)
        for label, count in zip(pred_unique, pred_counts)
    }

    error_pairs = []
    for true_label in range(num_classes):
        for pred_label in range(num_classes):
            if true_label == pred_label:
                continue
            count = int(cm[true_label, pred_label])
            if count > 0:
                error_pairs.append((count, true_label, pred_label))
    error_pairs.sort(reverse=True)

    analysis = {
        "checkpoint": str(checkpoint),
        "num_samples": int(len(labels)),
        "max_len": int(metadata["max_len"]),
        "model_input_len": int(MODEL_MAX_LEN),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "abs_sum": float(abs_sum),
        "mean_abs_sum": float(mean_abs_sum),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "true_distribution": true_distribution,
        "pred_distribution": pred_distribution,
        "confusion_matrix": cm.astype(int).tolist(),
        "top_error_pairs": [
            {"count": count, "true_label": true_label, "pred_label": pred_label}
            for count, true_label, pred_label in error_pairs[:10]
        ],
    }

    lines = [
        "新闻文本分类推理结果分析",
        "",
        f"模型权重: {checkpoint}",
        f"测试样本数: {len(labels)}",
        f"固定序列长度: {metadata['max_len']}",
        f"模型输入长度: {MODEL_MAX_LEN}",
        f"Accuracy: {accuracy:.4f}",
        f"Macro Precision: {macro_precision:.4f}",
        f"Macro Recall: {macro_recall:.4f}",
        f"Macro F1: {macro_f1:.4f}",
        f"Weighted F1: {weighted_f1:.4f}",
        f"ABS-SUM: {abs_sum:.4f}",
        f"Mean ABS-SUM: {mean_abs_sum:.6f}",
        f"平均置信度: {confidence.mean():.4f}",
        "",
        "分类报告",
        report,
        "",
        "混淆矩阵",
        str(cm),
        "",
        "主要错分方向",
    ]
    if error_pairs:
        for count, true_label, pred_label in error_pairs[:10]:
            lines.append(f"true={true_label} -> pred={pred_label}: {count}")
    else:
        lines.append("无错分样本")

    return "\n".join(lines) + "\n", analysis


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_ids, labels, ids, test_idx, metadata = load_test_split()
    num_classes = int(metadata["num_classes"])
    vocab_size = int(metadata["vocab_size"])

    save_test_set(args.output_dir, input_ids, labels, ids, test_idx, metadata)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NewsDataset(input_ids, labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = TextCNN(vocab_size=vocab_size, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    preds, probs, confidence = run_inference(model, loader, device)
    _, wrong_df = save_prediction_files(
        args.output_dir,
        ids,
        labels,
        preds,
        probs,
        confidence,
        num_classes,
    )

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    plot_confusion_matrix(args.output_dir, cm, num_classes)

    analysis_text, analysis = build_analysis(
        labels,
        preds,
        probs,
        confidence,
        cm,
        args.checkpoint,
        metadata,
    )
    (args.output_dir / "analysis.txt").write_text(analysis_text, encoding="utf-8")
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("========== 推理完成 ==========")
    print(f"测试集样本数: {len(labels)}")
    print(f"错分样本数: {len(wrong_df)}")
    print(f"预测明细: {args.output_dir / 'predictions.csv'}")
    print(f"提交概率: {args.output_dir / 'submit_result.csv'}")
    print(f"结果分析: {args.output_dir / 'analysis.txt'}")
    print(f"混淆矩阵: {args.output_dir / 'inference_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
