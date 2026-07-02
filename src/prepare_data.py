# -*- coding: utf-8 -*-
"""
新闻文本分类数据预处理

1. 读取 data/train_set.csv
2. 将空格分隔的词编号转为固定长度 token 序列
3. 保存 labels、ids、lengths、input_ids 和元信息
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "train_set.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 512
PAD_ID = 0
TOKEN_OFFSET = 1


def parse_text(text: str, max_len: int) -> tuple[np.ndarray, int, int]:
    tokens = [int(token) + TOKEN_OFFSET for token in str(text).split()]
    original_length = len(tokens)
    truncated_length = min(original_length, max_len)

    input_ids = np.full(max_len, PAD_ID, dtype=np.uint16)
    if truncated_length > 0:
        input_ids[:truncated_length] = np.asarray(
            tokens[:truncated_length],
            dtype=np.uint16,
        )

    max_token = max(tokens) if tokens else PAD_ID
    return input_ids, original_length, max_token


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"未找到原始数据文件：{CSV_PATH}")

    print(f"正在读取数据：{CSV_PATH}")
    df = pd.read_csv(CSV_PATH, sep="\t")

    required_columns = {"label", "text"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"数据必须包含列：{required_columns}")

    labels = df["label"].astype(np.int64).to_numpy()
    ids = np.arange(len(df), dtype=np.int64)

    input_ids = np.zeros((len(df), MAX_LEN), dtype=np.uint16)
    lengths = np.zeros(len(df), dtype=np.int32)
    max_token_id = 0

    print(f"原始样本数：{len(df)}")
    print(f"固定序列长度：{MAX_LEN}")

    for index, text in enumerate(tqdm(df["text"], desc="Parsing texts")):
        row_ids, original_length, row_max_token = parse_text(text, MAX_LEN)
        input_ids[index] = row_ids
        lengths[index] = original_length
        max_token_id = max(max_token_id, row_max_token)

    vocab_size = int(max_token_id + 1)
    num_classes = int(labels.max() + 1)

    np.save(OUTPUT_DIR / "input_ids.npy", input_ids)
    np.save(OUTPUT_DIR / "labels.npy", labels)
    np.save(OUTPUT_DIR / "ids.npy", ids)
    np.save(OUTPUT_DIR / "lengths.npy", lengths)

    metadata = {
        "max_len": MAX_LEN,
        "pad_id": PAD_ID,
        "token_offset": TOKEN_OFFSET,
        "vocab_size": vocab_size,
        "num_classes": num_classes,
        "num_samples": int(len(df)),
        "csv_path": str(CSV_PATH),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== 预处理完成 ==========")
    print(f"input_ids 形状：{input_ids.shape}")
    print(f"标签数量：{labels.shape}")
    print(f"类别数：{num_classes}")
    print(f"词表大小：{vocab_size}")
    print(f"文本长度均值：{lengths.mean():.2f}")
    print(f"文本长度中位数：{np.median(lengths):.0f}")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
