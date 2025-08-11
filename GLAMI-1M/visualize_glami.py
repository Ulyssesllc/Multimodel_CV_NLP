import os
import argparse
import json
import textwrap
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

from Contrastive import Q_cons_fusion
from MLP import MLP_fusion
from process_data import MyData  # assumes accessible (as used in train.py)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Visualize GLAMI-1M model predictions")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth",
        help="Path to checkpoint or raw state dict",
    )
    p.add_argument(
        "--model-type",
        choices=["Q_cons_fusion", "MLP_fusion"],
        default="Q_cons_fusion",
        help="Model architecture",
    )
    p.add_argument(
        "--train-csv",
        default="GLAMI-1M-train.csv",
        help="Train CSV (for label map order)",
    )
    p.add_argument(
        "--test-csv", default="GLAMI-1M-test.csv", help="Test CSV to sample from"
    )
    p.add_argument("--images-dir", default="images", help="Images directory")
    p.add_argument(
        "--num-samples", type=int, default=6, help="Number of samples to visualize"
    )
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for sampling")
    p.add_argument("--top-k", type=int, default=5, help="Top-K predictions to show")
    p.add_argument(
        "--output-dir", default="glami_vis", help="Directory to save outputs"
    )
    p.add_argument(
        "--composite",
        action="store_true",
        help="Create single composite PNG with panels",
    )
    p.add_argument(
        "--layout",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Composite stacking direction",
    )
    p.add_argument(
        "--no-grid", action="store_true", help="Skip matplotlib grid (only composite)"
    )
    p.add_argument(
        "--no-show", action="store_true", help="Headless mode (don't open windows)"
    )
    # parity options with amazon visualize
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature scaling for softmax (display only)",
    )
    p.add_argument(
        "--max-desc-chars",
        type=int,
        default=200,
        help="Truncate description text before wrapping",
    )
    p.add_argument(
        "--embed-json",
        action="store_true",
        help="Embed JSON summary panel into composite image",
    )
    p.add_argument(
        "--json-wrap",
        type=int,
        default=110,
        help="Approx char wrap width for embedded JSON",
    )
    p.add_argument(
        "--no-json-file",
        action="store_true",
        help="Do not emit separate JSON file when set",
    )
    return p


def load_label_map(train_csv: str) -> Dict[str, int]:
    import pandas as pd

    df = pd.read_csv(train_csv)
    return {label: idx for idx, label in enumerate(df["category_name"].unique())}


def instantiate_model(model_type: str, num_classes: int):
    if model_type == "Q_cons_fusion":
        model = Q_cons_fusion()
        # Adjust final fusion linear for classes if mismatch
        if getattr(model.fusion[-1], "out_features", None) != num_classes:
            # model.fusion last layer is Linear(256,191); replace if needed
            from torch import nn

            layers = list(model.fusion.children())
            # replace last layer
            if isinstance(layers[-1], nn.Linear):
                in_f = layers[-1].in_features
                layers[-1] = nn.Linear(in_f, num_classes)
                model.fusion = nn.Sequential(*layers)
        return model
    else:
        model = MLP_fusion()
        # Replace final_fc if shape mismatch
        if model.final_fc.out_features != num_classes:
            in_f = model.final_fc.in_features
            model.final_fc = torch.nn.Linear(in_f, num_classes)
        return model


def strip_module_prefix(state_dict: Dict[str, Any]):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_model(model, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def tensor_to_pil(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img_np = (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img_np * 255).astype("uint8")
    return Image.fromarray(img)


def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(args.train_csv)
    # reverse map
    label_map_inv = {v: k for k, v in label_map.items()}

    # Build dataset using existing MyData interface
    test_dataset = MyData(args.test_csv, args.images_dir, label_map)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Infer num_classes from label_map size
    num_classes = len(label_map)
    model = instantiate_model(args.model_type, num_classes)
    model = load_checkpoint_model(model, args.checkpoint)

    samples = []

    # Optionally build matplotlib grid
    show_grid = not args.no_grid
    if show_grid:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, args.num_samples, figsize=(4 * args.num_samples, 7))
        fig.subplots_adjust(bottom=0.45)
    else:
        fig = None
        axes = None

    collected = 0
    for batch in loader:
        if collected >= args.num_samples:
            break
        img = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].item() if img.size(0) == 1 else batch["label"][0].item()
        with torch.no_grad():
            if args.model_type == "Q_cons_fusion":
                logits, img_feat, text_feat = model(img, input_ids, attention_mask)
            else:
                logits = model(img, input_ids, attention_mask)
            temperature = (
                args.temperature if args.temperature and args.temperature > 0 else 1.0
            )
            probs = F.softmax(logits / temperature, dim=1).squeeze(0).cpu()
        pred_idx = int(torch.argmax(probs).item())
        k = min(args.top_k, probs.shape[0])
        topk_vals, topk_indices = torch.topk(probs, k)
        topk = [
            {
                "rank": r + 1,
                "label_index": int(idx.item()),
                "label_name": label_map_inv.get(int(idx.item()), str(int(idx.item()))),
                "prob": float(val.item()),
            }
            for r, (idx, val) in enumerate(zip(topk_indices, topk_vals))
        ]
        gt_name = label_map_inv.get(label, str(label))
        pred_name = label_map_inv.get(pred_idx, str(pred_idx))
        desc = f"Category: {gt_name}"
        if args.max_desc_chars and len(desc) > args.max_desc_chars:
            desc = desc[: args.max_desc_chars] + "..."
        samples.append(
            {
                "sample_index": collected,
                "ground_truth": {"index": int(label), "name": gt_name},
                "prediction": {"index": pred_idx, "name": pred_name},
                "topk": topk,
                "description": desc,
                "image_tensor": batch["image"].squeeze(0).cpu(),
            }
        )
        if show_grid:
            from amazon.visualize_results import imshow  # reuse helper

            ax = axes[collected] if args.num_samples > 1 else axes
            imshow(batch["image"].squeeze(0), ax)
            wrapped_desc = textwrap.fill(desc, width=50)
            caption = "\n".join(
                [
                    f"GT: {gt_name} ({label}) | Pred: {pred_name} ({pred_idx})",
                    "TopK: "
                    + ", ".join([f"{t['label_name']}({t['prob']:.2f})" for t in topk]),
                    "Desc: " + wrapped_desc,
                ]
            )
            ax.set_xlabel(caption, fontsize=7)
        collected += 1

    os.makedirs(args.output_dir, exist_ok=True)
    if show_grid:
        import matplotlib.pyplot as plt

        grid_path = os.path.join(args.output_dir, "glami_predictions_grid.png")
        plt.tight_layout()
        plt.savefig(grid_path, dpi=200)
        print(f"Saved grid visualization to {grid_path}")
        if not args.no_show and not args.composite:
            plt.show()
        else:
            plt.close()

    # Composite single file
    if args.composite:
        panels = []
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for s in samples:
            img_tensor = s["image_tensor"]
            pil_img = tensor_to_pil(img_tensor)
            W, H = pil_img.size
            topk_line = "TopK: " + ", ".join(
                [f"{t['label_name']}({t['prob']:.2f})" for t in s["topk"]]
            )
            topk_wrapped = textwrap.wrap(topk_line, width=70)
            desc_wrapped = textwrap.wrap("Desc: " + s["description"], width=70)
            header_lines = [
                f"Sample {s['sample_index']}",
                f"GT: {s['ground_truth']['name']} ({s['ground_truth']['index']})",
                f"Pred: {s['prediction']['name']} ({s['prediction']['index']})",
            ]
            lines = header_lines + topk_wrapped + desc_wrapped
            if font is None:
                line_h = 14

                def estimate_w(txt):
                    return len(txt) * 6
            else:
                line_h = font.getbbox("A")[3]

                def estimate_w(txt):
                    b = font.getbbox(txt if txt else "A")
                    return b[2] - b[0]

            text_h = line_h * len(lines) + 10
            max_text_w = max(estimate_w(t) for t in lines) + 10
            panel_w = max(W, max_text_w)
            panel = Image.new("RGB", (panel_w, H + text_h), (255, 255, 255))
            x_img = (panel_w - W) // 2
            panel.paste(pil_img, (x_img, 0))
            draw = ImageDraw.Draw(panel)
            y = H + 5
            for ln in lines:
                draw.text((5, y), ln, fill=(0, 0, 0), font=font)
                y += line_h
            draw.rectangle(
                [0, 0, panel_w - 1, H + text_h - 1], outline=(0, 0, 0), width=1
            )
            panels.append(panel)
        if args.layout == "vertical":
            total_h = sum(p.size[1] for p in panels) + (len(panels) - 1) * 8
            max_w = max(p.size[0] for p in panels)
            composite = Image.new("RGB", (max_w, total_h), (245, 245, 245))
            y = 0
            for p in panels:
                composite.paste(p, (0, y))
                y += p.size[1] + 8
        else:
            total_w = sum(p.size[0] for p in panels) + (len(panels) - 1) * 8
            max_h = max(p.size[1] for p in panels)
            composite = Image.new("RGB", (total_w, max_h), (245, 245, 245))
            x = 0
            for p in panels:
                composite.paste(p, (x, 0))
                x += p.size[0] + 8
        # Optional embedded JSON
        if args.embed_json:
            summary_obj = {"samples": samples}
            raw_json = json.dumps(summary_obj, ensure_ascii=False, indent=2)
            wrapped_lines = []
            for ln in raw_json.splitlines():
                if len(ln) <= args.json_wrap:
                    wrapped_lines.append(ln)
                else:
                    s = ln
                    while len(s) > args.json_wrap:
                        wrapped_lines.append(s[: args.json_wrap])
                        s = s[args.json_wrap :]
                    if s:
                        wrapped_lines.append(s)
            try:
                font2 = ImageFont.load_default()
            except Exception:
                font2 = font
            font_use = font2 or font
            line_h2 = font_use.getbbox("A")[3] if font_use else 14
            pad = 6
            est_char_w = 6
            panel_w_json = max(len(t) for t in wrapped_lines) * est_char_w + pad * 2
            panel_h_json = line_h2 * len(wrapped_lines) + pad * 2
            json_panel = Image.new("RGB", (panel_w_json, panel_h_json), (255, 255, 255))
            djson = ImageDraw.Draw(json_panel)
            yj = pad
            for ln in wrapped_lines:
                djson.text((pad, yj), ln, fill=(0, 0, 0), font=font_use)
                yj += line_h2
            if args.layout == "vertical":
                new_h = composite.size[1] + panel_h_json + 8
                new_w = max(composite.size[0], panel_w_json)
                extended = Image.new("RGB", (new_w, new_h), (235, 235, 235))
                extended.paste(composite, (0, 0))
                extended.paste(json_panel, (0, composite.size[1] + 8))
            else:
                new_w = composite.size[0] + panel_w_json + 8
                new_h = max(composite.size[1], panel_h_json)
                extended = Image.new("RGB", (new_w, new_h), (235, 235, 235))
                extended.paste(composite, (0, 0))
                extended.paste(json_panel, (composite.size[0] + 8, 0))
            composite = extended

        comp_path = os.path.join(args.output_dir, "glami_predictions_composite.png")
        composite.save(comp_path)
        print(f"Saved composite visualization to {comp_path}")
        if not args.no_show:
            try:
                composite.show()
            except Exception:
                pass

    # JSON export
    if not args.no_json_file:
        json_path = os.path.join(args.output_dir, "glami_predictions.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)
        print(f"Saved structured predictions to {json_path}")

    # Console summary
    for s in samples:
        print("--- Sample", s["sample_index"], "---")
        print("GT:", s["ground_truth"]["name"], "| Pred:", s["prediction"]["name"])
        print(
            "TopK:",
            ", ".join([f"{t['label_name']}({t['prob']:.2f})" for t in s["topk"]]),
        )
        print("Desc:", s["description"])


def main():
    args = build_arg_parser().parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
