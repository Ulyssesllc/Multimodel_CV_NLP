"""Console visualization for GLAMI-1M models (Q_cons_fusion / MLP_fusion).

Changes (parity with amazon/visualize_results.py console-only mode):
 - Removed matplotlib grid + JSON export (no files are written by default).
 - Added optional ASCII preview (grayscale) or ANSI truecolor block preview ("--color-ascii").
 - Prints per-sample block: optional ASCII image, Sample header, GT | Pred, TopK, Text (token source).
 - Deterministic sampling via --seed, strategy random/first.

Example:
    python visualize_glami.py \
        --checkpoint checkpoints/best_model.pth \
        --model-type Q_cons_fusion \
        --train-csv GLAMI-1M-train.csv \
        --test-csv GLAMI-1M-test.csv \
        --images images \
        --num-samples 6 --top-k 5 --ascii-preview --color-ascii

Notes:
 - Q_cons_fusion path is primary; MLP_fusion loaded best-effort.
 - Text column (default category_name) is tokenized for models needing text; also echoed in console.
 - To restore JSON/grid saving, extend this script (see git history for previous version).
"""

from __future__ import annotations
import os
import random
import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import BertTokenizer

from Contrastive import Q_cons_fusion

try:
    from MLP import MLP_fusion  # type: ignore
except Exception:
    MLP_fusion = None  # fallback if import fails


def parse_args():
    p = argparse.ArgumentParser(
        description="Console visualize GLAMI-1M predictions (no file outputs)"
    )
    p.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.pth)"
    )
    p.add_argument(
        "--model-type", choices=["Q_cons_fusion", "MLP_fusion"], default="Q_cons_fusion"
    )
    p.add_argument("--train-csv", default="GLAMI-1M-train.csv")
    p.add_argument("--test-csv", default="GLAMI-1M-test.csv")
    p.add_argument("--images", default="images", help="Directory with image files")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--text-column",
        default="category_name",
        help="Column used as text input where needed",
    )
    p.add_argument("--sample-strategy", choices=["random", "first"], default="random")
    # Console visualization flags (mirroring amazon script)
    p.add_argument(
        "--ascii-preview",
        action="store_true",
        help="Print ASCII art preview for each image",
    )
    p.add_argument(
        "--color-ascii",
        action="store_true",
        help="Use ANSI truecolor blocks for preview (needs 24-bit terminal)",
    )
    p.add_argument(
        "--ascii-width",
        type=int,
        default=40,
        help="Width (characters) for ASCII preview",
    )
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_label_map(train_csv: str) -> Dict[str, int]:
    df = pd.read_csv(train_csv)
    labels = list(df["category_name"].unique())
    return {lab: i for i, lab in enumerate(labels)}


def load_samples(test_csv: str, num: int, strategy: str) -> pd.DataFrame:
    df = pd.read_csv(test_csv)
    if strategy == "first" or len(df) <= num:
        return df.head(num).copy()
    return df.sample(n=num, random_state=42).copy()


def load_image(path: str, transform):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (224, 224), (0, 0, 0))
    return transform(img)


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def strip_module(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module.") :]] = v
        else:
            new_sd[k] = v
    return new_sd


def load_model(args, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "Q_cons_fusion":
        model = Q_cons_fusion()
    else:
        if MLP_fusion is None:
            raise RuntimeError(
                "MLP_fusion not available; cannot visualize this model type."
            )
        model = MLP_fusion()
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("model_state", ckpt)
    sd = strip_module(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(
            f"[Warn] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    model.eval().to(device)
    return model, device


def predict(model, device, batch_imgs, batch_text_inputs, model_type):
    with torch.no_grad():
        if model_type == "Q_cons_fusion":
            logits, _, _ = model(
                batch_imgs,
                batch_text_inputs["input_ids"],
                batch_text_inputs["attention_mask"],
            )
        else:
            try:
                logits = model(
                    batch_imgs,
                    batch_text_inputs["input_ids"],
                    batch_text_inputs["attention_mask"],
                )
            except Exception:
                logits = model(batch_imgs, batch_text_inputs["raw_text"])  # type: ignore
    return F.softmax(logits, dim=1)


############################################################
# ASCII RENDER HELPERS (mirroring amazon console script)    #
############################################################
def tensor_to_ascii(t: torch.Tensor, width: int = 40) -> str:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = (t.cpu() * std + mean).clamp(0, 1)
    g = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    h, w = g.shape
    aspect = h / w
    new_w = width
    new_h = max(1, int(aspect * new_w * 0.55))
    g_resized = torch.nn.functional.interpolate(
        g.unsqueeze(0).unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    chars = " .:-=+*#%@"
    idx = (g_resized * (len(chars) - 1)).round().long()
    lines = ["".join(chars[i] for i in row) for row in idx]
    return "\n" + "\n".join(lines)


def tensor_to_color_ascii(t: torch.Tensor, width: int = 40) -> str:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = (t.cpu() * std + mean).clamp(0, 1)
    h, w = x.shape[1:]
    aspect = h / w
    new_w = width
    new_h = max(1, int(aspect * new_w * 0.55))
    x_resized = torch.nn.functional.interpolate(
        x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    lines: List[str] = []
    for row in range(new_h):
        parts = []
        for col in range(new_w):
            r = int(x_resized[0, row, col].item() * 255)
            g = int(x_resized[1, row, col].item() * 255)
            b = int(x_resized[2, row, col].item() * 255)
            parts.append(f"\x1b[48;2;{r};{g};{b}m ")
        parts.append("\x1b[0m")
        lines.append("".join(parts))
    return "\n" + "\n".join(lines)


def main():
    args = parse_args()
    set_seed(args.seed)

    label_map = build_label_map(args.train_csv)
    label_map_inv = {v: k for k, v in label_map.items()}
    samples = load_samples(args.test_csv, args.num_samples, args.sample_strategy)

    transform = build_transform()
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    images = []
    texts: List[str] = []
    for _, row in samples.iterrows():
        img_path = (
            os.path.join(args.images, row["img_path"])
            if not os.path.isfile(row["img_path"])
            else row["img_path"]
        )
        images.append(load_image(img_path, transform))
        texts.append(str(row.get(args.text_column, "")))

    batch_imgs = torch.stack(images)
    tokenized = tokenizer(
        texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    tokenized["raw_text"] = texts

    model, device = load_model(args, num_classes=len(label_map))
    batch_imgs = batch_imgs.to(device)
    tokenized = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tokenized.items()
    }
    probs = predict(model, device, batch_imgs, tokenized, args.model_type)

    warned_color = False
    for i, (_idx, row) in enumerate(samples.iterrows()):
        p = probs[i]
        topv, topi = p.topk(min(args.top_k, p.size(0)))
        topk_pairs = list(zip(topv.tolist(), topi.tolist()))
        pred_idx = int(topi[0])
        gt = row["category_name"]
        pred_name = label_map_inv.get(pred_idx, str(pred_idx))
        topk_str = ", ".join(
            [
                f"{label_map_inv.get(idx, str(idx))}({val:.2f})"
                for val, idx in topk_pairs
            ]
        )
        if args.ascii_preview:
            if args.color_ascii:
                if not warned_color:
                    print(
                        "[Info] Using ANSI truecolor blocks; ensure terminal supports 24-bit color."
                    )
                    warned_color = True
                print(tensor_to_color_ascii(batch_imgs[i].cpu(), args.ascii_width))
            else:
                print(tensor_to_ascii(batch_imgs[i].cpu(), args.ascii_width))
        print(f"--- Sample {i} ---")
        print(f"GT | Pred: {gt} | {pred_name}")
        print(f"TopK: {topk_str}")
        txt = texts[i]
        trunc_txt = txt[:300] + ("..." if len(txt) > 300 else "")
        print(f"Text: {trunc_txt}")


if __name__ == "__main__":
    main()
