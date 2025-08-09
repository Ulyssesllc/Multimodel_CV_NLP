"""Utility script to prepare and validate the GLAMI-1M dataset.

Features:
- Stratified train/test split from a master metadata CSV.
- Validation of image existence.
- Optional lowercase normalization of filenames.
- Summary statistics export (class counts, missing files, top-N classes, etc.).

Expected input master CSV schema (minimum columns):
    img_path,category_name

Usage examples:
    python prepare_glami.py --meta all_meta.csv --images images
    python prepare_glami.py --meta all_meta.csv --images images --test-size 0.1 --seed 42
    python prepare_glami.py --meta all_meta.csv --images images --lowercase --summary summary.json

Outputs:
    GLAMI-1M-train.csv
    GLAMI-1M-test.csv
    (optional) summary.json
"""

from __future__ import annotations
import os
import argparse
import json
from collections import Counter
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description="Prepare GLAMI-1M dataset")
    p.add_argument("--meta", required=True, help="Path to master metadata CSV")
    p.add_argument("--images", required=True, help="Directory containing image files")
    p.add_argument("--test-size", type=float, default=0.1, help="Test split ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument(
        "--train-out", default="GLAMI-1M-train.csv", help="Train CSV output filename"
    )
    p.add_argument(
        "--test-out", default="GLAMI-1M-test.csv", help="Test CSV output filename"
    )
    p.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase filenames and update img_path",
    )
    p.add_argument(
        "--summary", default=None, help="Optional path to write JSON summary stats"
    )
    return p.parse_args()


def load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"img_path", "category_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata: {missing}")
    return df


def maybe_lowercase_files(df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
    if not any(ch.isupper() for ch in "".join(df["img_path"].astype(str))):
        return df  # nothing to do
    mapping = {}
    for original in df["img_path"]:
        src = os.path.join(images_dir, original)
        lower_name = original.lower()
        dst = os.path.join(images_dir, lower_name)
        if os.path.isfile(src) and src != dst:
            try:
                os.rename(src, dst)
            except OSError:
                pass
        mapping[original] = lower_name
    df["img_path"] = df["img_path"].map(mapping).fillna(df["img_path"])
    return df


def validate_files(df: pd.DataFrame, images_dir: str) -> Tuple[pd.DataFrame, list]:
    exists_mask = df["img_path"].apply(
        lambda p: os.path.isfile(os.path.join(images_dir, p))
    )
    missing = df.loc[~exists_mask, "img_path"].tolist()
    kept = df.loc[exists_mask].reset_index(drop=True)
    return kept, missing


def stratified_split(df: pd.DataFrame, test_size: float, seed: int):
    return train_test_split(
        df,
        test_size=test_size,
        stratify=df["category_name"],
        random_state=seed,
        shuffle=True,
    )


def build_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, missing: list) -> dict:
    train_counts = Counter(train_df["category_name"])
    test_counts = Counter(test_df["category_name"])
    return {
        "num_train": len(train_df),
        "num_test": len(test_df),
        "num_classes": train_df["category_name"].nunique(),
        "top5_train_classes": train_counts.most_common(5),
        "top5_test_classes": test_counts.most_common(5),
        "missing_files": missing[:50],  # show first 50 only
        "num_missing_files": len(missing),
    }


def main():
    args = parse_args()
    df = load_metadata(args.meta)

    if args.lowercase:
        df = maybe_lowercase_files(df, args.images)

    df_valid, missing = validate_files(df, args.images)
    if not df_valid.empty:
        train_df, test_df = stratified_split(df_valid, args.test_size, args.seed)
    else:
        raise RuntimeError("No valid image files found after validation.")

    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)

    print(
        f"Saved train -> {args.train_out} ({len(train_df)}) | test -> {args.test_out} ({len(test_df)})"
    )
    if missing:
        print(
            f"Warning: {len(missing)} files referenced but not found (showing first 10): {missing[:10]}"
        )

    if args.summary:
        summary = build_summary(train_df, test_df, missing)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Wrote summary -> {args.summary}")


if __name__ == "__main__":
    main()
