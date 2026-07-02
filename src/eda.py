# -*- coding: utf-8 -*-
"""
新闻文本分类 EDA

绘制类别分布和文本长度分布。
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    labels_path = PROCESSED_DIR / "labels.npy"
    lengths_path = PROCESSED_DIR / "lengths.npy"
    metadata_path = PROCESSED_DIR / "metadata.json"

    if not labels_path.exists() or not lengths_path.exists():
        raise FileNotFoundError("请先运行：python src/prepare_data.py")

    labels = np.load(labels_path)
    lengths = np.load(lengths_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    max_len = int(metadata["max_len"])

    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(9, 5))
    plt.bar(unique, counts)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("News Dataset Label Distribution")
    plt.xticks(unique)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "label_distribution.png", dpi=300)
    plt.close()

    clipped_lengths = np.minimum(lengths, 3000)
    plt.figure(figsize=(9, 5))
    plt.hist(clipped_lengths, bins=60)
    plt.axvline(max_len, color="red", linestyle="--", label=f"Max Len = {max_len}")
    plt.xlabel("Text Length")
    plt.ylabel("Count")
    plt.title("News Dataset Text Length Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "text_length_distribution.png", dpi=300)
    plt.close()

    summary = (
        "新闻文本数据探索\n"
        f"样本数: {len(labels)}\n"
        f"类别数: {len(unique)}\n"
        f"固定序列长度: {max_len}\n"
        f"长度均值: {lengths.mean():.2f}\n"
        f"长度中位数: {np.median(lengths):.0f}\n"
        f"长度 P90: {np.quantile(lengths, 0.90):.0f}\n"
        f"长度 P95: {np.quantile(lengths, 0.95):.0f}\n"
        f"长度 P99: {np.quantile(lengths, 0.99):.0f}\n"
        f"最大长度: {lengths.max()}\n\n"
        "类别分布\n"
    )
    for label, count in zip(unique, counts):
        summary += f"label={label}: {count} ({count / len(labels) * 100:.2f}%)\n"

    (FIGURE_DIR / "eda_summary.txt").write_text(summary, encoding="utf-8")

    print("========== EDA 完成 ==========")
    print(f"类别分布图：{FIGURE_DIR / 'label_distribution.png'}")
    print(f"长度分布图：{FIGURE_DIR / 'text_length_distribution.png'}")
    print(f"统计摘要：{FIGURE_DIR / 'eda_summary.txt'}")


if __name__ == "__main__":
    main()
