# -*- coding: utf-8 -*-
"""
新闻文本分类数据划分

按 8:2 分层划分训练集和测试集，并保存索引与样本划分记录。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
TEST_SIZE = 0.2

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    labels_path = PROCESSED_DIR / "labels.npy"
    ids_path = PROCESSED_DIR / "ids.npy"

    if not labels_path.exists() or not ids_path.exists():
        raise FileNotFoundError("请先运行：python src/prepare_data.py")

    labels = np.load(labels_path)
    ids = np.load(ids_path)
    indices = np.arange(len(labels))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    np.savez(
        OUTPUT_DIR / "split_indices.npz",
        train_idx=train_idx,
        test_idx=test_idx,
    )

    split_labels = np.full(len(labels), "test", dtype="<U5")
    split_labels[train_idx] = "train"
    split_df = pd.DataFrame({
        "id": ids,
        "set": split_labels,
        "label": labels,
    })
    split_df.to_csv(OUTPUT_DIR / "split_ids.csv", index=False, encoding="utf-8-sig")

    print("========== 数据划分完成 ==========")
    print(f"随机种子：{SEED}")
    print(f"训练集样本数：{len(train_idx)}")
    print(f"测试集样本数：{len(test_idx)}")

    for name, idx in [("TrainSet", train_idx), ("TestSet", test_idx)]:
        print(f"\n{name} 类别分布：")
        unique, counts = np.unique(labels[idx], return_counts=True)
        for label, count in zip(unique, counts):
            print(f"label={label}: {count} ({count / len(idx) * 100:.2f}%)")


if __name__ == "__main__":
    main()
