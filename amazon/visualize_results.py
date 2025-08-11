import os
import json
import textwrap
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from process_data import MyData
from MLP import FusionModule


def imshow(img_tensor, ax):
    """
    Unnormalize and show a tensor image on a matplotlib axis.
    """
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img = img_tensor.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.axis("off")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize VLM model outputs")
    parser.add_argument(
        "--model-path", default="fusion_model_best.pth", help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of samples to display"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for loading samples"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Top-K predictions to record"
    )
    parser.add_argument(
        "--output-dir", default="vlm_outputs", help="Directory to store outputs"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open window (headless)"
    )
    parser.add_argument(
        "--composite",
        action="store_true",
        help="Create one single composite image containing all samples with text inside each frame",
    )
    parser.add_argument(
        "--layout",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Panel stacking direction for composite output",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip legacy matplotlib grid image generation",
    )
    # new options
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature scaling for softmax (display only)",
    )
    parser.add_argument(
        "--max-desc-chars",
        type=int,
        default=300,
        help="Truncate description before wrapping",
    )
    parser.add_argument(
        "--embed-json",
        action="store_true",
        help="Embed JSON summary panel into composite PNG",
    )
    parser.add_argument(
        "--json-wrap",
        type=int,
        default=110,
        help="Wrap width (approx chars) for embedded JSON",
    )
    parser.add_argument(
        "--no-json-file",
        action="store_true",
        help="Do not write separate predictions.json file",
    )
    return parser


def visualize(
    num_samples=5,
    batch_size=1,
    model_path="fusion_model_best.pth",
    top_k=5,
    output_dir="vlm_outputs",
    show=True,
    composite=False,
    layout="vertical",
    skip_grid=False,
    temperature=1.0,
    max_desc_chars=300,
    embed_json=False,
    json_wrap=110,
    no_json_file=False,
):
    # Setup dataset
    base_dir = os.path.dirname(__file__)
    csv_file = os.path.join(base_dir, "training_data.csv")
    img_dir = os.path.join(base_dir, "amazon_dataset")
    dataset = MyData(csv_file, img_dir)
    label_map_inv = {v: k for k, v in dataset.label_map.items()}

    os.makedirs(os.path.join(base_dir, output_dir), exist_ok=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    num_classes_checkpoint = state_dict["final_fc.weight"].shape[0]
    model = FusionModule(num_classes=num_classes_checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    samples_collected = []

    # Prepare matplotlib grid only if requested
    if not skip_grid:
        num_cols = num_samples
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 7))
        fig.subplots_adjust(bottom=0.45)  # more space for captions
    else:
        fig = None
        axes = None

    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        img = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_idx = batch["label"].item()
        desc = batch.get("description")
        if isinstance(desc, (list, tuple)):
            desc = desc[0]
        if desc is None or str(desc).lower() == "nan":
            desc = "(no description)"
        if max_desc_chars and len(desc) > max_desc_chars:
            desc = desc[:max_desc_chars] + "..."

        with torch.no_grad():
            logits = model(img, input_ids, attention_mask)
            if temperature <= 0:
                temperature = 1.0
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=1).cpu().squeeze(0)
            pred_idx = int(torch.argmax(probs).item())

        # Top-K predictions
        k = min(top_k, probs.shape[0])
        topk_vals, topk_indices = torch.topk(probs, k)
        topk = [
            {
                "rank": rank + 1,
                "label_index": int(idx.item()),
                "label_name": label_map_inv.get(int(idx.item()), str(int(idx.item()))),
                "prob": float(val.item()),
            }
            for rank, (idx, val) in enumerate(zip(topk_indices, topk_vals))
        ]

        gt_name = label_map_inv.get(label_idx, str(label_idx))
        pred_name = label_map_inv.get(pred_idx, str(pred_idx))

        samples_collected.append(
            {
                "sample_index": i,
                "ground_truth": {"index": int(label_idx), "name": gt_name},
                "prediction": {"index": pred_idx, "name": pred_name},
                "topk": topk,
                "description": desc,
                # store the actual image tensor used so we don't re-fetch with a different order
                "image_tensor": batch["image"].squeeze(0).cpu(),
            }
        )

        if not skip_grid:
            ax = axes[i] if num_cols > 1 else axes
            imshow(batch["image"].squeeze(0), ax)
            wrapped_desc = textwrap.fill(desc, width=50)
            caption_lines = [
                f"GT: {gt_name} ({label_idx}) | Pred: {pred_name} ({pred_idx})",
                "TopK: "
                + ", ".join([f"{t['label_name']}({t['prob']:.2f})" for t in topk]),
                "Desc: " + wrapped_desc,
            ]
            caption = "\n".join(caption_lines)
            ax.set_xlabel(caption, fontsize=7)

    if not skip_grid:
        for j in range(i + 1, num_cols):
            ax_empty = axes[j] if num_cols > 1 else axes
            ax_empty.axis("off")
        plt.tight_layout()
        save_path = os.path.join(base_dir, output_dir, "predictions_grid.png")
        plt.savefig(save_path, dpi=200)
        print(f"Saved visualization grid to {save_path}")
        if show and not composite:
            plt.show()
        else:
            plt.close(fig)

    # Composite single-file output
    if composite:
        comp_path = os.path.join(base_dir, output_dir, "predictions_composite.png")
        panel_images = []
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        def tensor_to_pil(t):
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            img_np = (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            img = (img_np * 255).astype("uint8")
            return Image.fromarray(img)

        max_img_w = 0
        # Build per-sample panels
        for sample in samples_collected:
            # Use stored tensor to ensure consistency with predictions (avoid shuffle mismatch)
            img_tensor = sample["image_tensor"]
            pil_img = tensor_to_pil(img_tensor)
            W, H = pil_img.size
            max_img_w = max(max_img_w, W)
            # Prepare text
            topk_line = "TopK: " + ", ".join(
                [f"{t['label_name']}({t['prob']:.2f})" for t in sample["topk"]]
            )
            # Wrap long lines (TopK, Description)
            topk_wrapped = textwrap.wrap(topk_line, width=70)
            desc_wrapped = textwrap.wrap("Desc: " + sample["description"], width=70)
            header_lines = [
                f"Sample {sample['sample_index']}",
                f"GT: {sample['ground_truth']['name']} ({sample['ground_truth']['index']})",
                f"Pred: {sample['prediction']['name']} ({sample['prediction']['index']})",
            ]
            text_lines = header_lines + topk_wrapped + desc_wrapped
            # Measure text height
            if font is None:
                line_height = 14
            else:
                line_height = font.getbbox("A")[3]
            text_height = line_height * len(text_lines) + 10  # padding bottom

            # Determine required text width in pixels to avoid horizontal clipping
            if font is None:

                def estimate_w(s):
                    return len(s) * 6  # rough average width
            else:

                def estimate_w(s):
                    bbox = font.getbbox(s if s else "A")
                    return bbox[2] - bbox[0]

            max_text_w = (
                max([estimate_w(t) for t in text_lines]) + 10
            )  # padding left/right
            panel_w = max(W, max_text_w)
            panel = Image.new("RGB", (panel_w, H + text_height), (255, 255, 255))
            # Center the image if panel wider than image
            x_img = (panel_w - W) // 2
            panel.paste(pil_img, (x_img, 0))
            draw = ImageDraw.Draw(panel)
            y_text = H + 5
            for line in text_lines:
                draw.text((5, y_text), line, fill=(0, 0, 0), font=font)
                y_text += line_height
            # Border
            draw.rectangle(
                [0, 0, panel_w - 1, H + text_height - 1], outline=(0, 0, 0), width=1
            )
            panel_images.append(panel)

        # Stack panels
        if layout == "vertical":
            total_height = (
                sum(p.size[1] for p in panel_images) + (len(panel_images) - 1) * 8
            )
            composite_img = Image.new("RGB", (max_img_w, total_height), (245, 245, 245))
            y = 0
            for p in panel_images:
                composite_img.paste(p, (0, y))
                y += p.size[1] + 8
        else:  # horizontal
            total_width = (
                sum(p.size[0] for p in panel_images) + (len(panel_images) - 1) * 8
            )
            max_height = max(p.size[1] for p in panel_images)
            composite_img = Image.new("RGB", (total_width, max_height), (245, 245, 245))
            x = 0
            for p in panel_images:
                composite_img.paste(p, (x, 0))
                x += p.size[0] + 8

        if embed_json:
            summary_obj = {"samples": samples_collected}
            raw_json = json.dumps(summary_obj, ensure_ascii=False, indent=2)
            wrapped_lines = []
            for line in raw_json.splitlines():
                if len(line) <= json_wrap:
                    wrapped_lines.append(line)
                else:
                    s = line
                    while len(s) > json_wrap:
                        wrapped_lines.append(s[:json_wrap])
                        s = s[json_wrap:]
                    if s:
                        wrapped_lines.append(s)
            try:
                font2 = ImageFont.load_default()
            except Exception:
                font2 = font
            font_use = font2 or font
            line_h = font_use.getbbox("A")[3] if font_use else 14
            pad = 6
            est_char_w = 6
            panel_w = (
                max(len(line_txt) for line_txt in wrapped_lines) * est_char_w + pad * 2
            )
            panel_h = line_h * len(wrapped_lines) + pad * 2
            json_panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
            djson = ImageDraw.Draw(json_panel)
            yj = pad
            for line_txt in wrapped_lines:
                djson.text((pad, yj), line_txt, fill=(0, 0, 0), font=font_use)
                yj += line_h
            if layout == "vertical":
                new_h = composite_img.size[1] + panel_h + 8
                new_w = max(composite_img.size[0], panel_w)
                extended = Image.new("RGB", (new_w, new_h), (235, 235, 235))
                extended.paste(composite_img, (0, 0))
                extended.paste(json_panel, (0, composite_img.size[1] + 8))
            else:
                new_w = composite_img.size[0] + panel_w + 8
                new_h = max(composite_img.size[1], panel_h)
                extended = Image.new("RGB", (new_w, new_h), (235, 235, 235))
                extended.paste(composite_img, (0, 0))
                extended.paste(json_panel, (composite_img.size[0] + 8, 0))
            composite_img = extended

        composite_img.save(comp_path)
        print(f"Saved composite single-file visualization to {comp_path}")
        if show:
            try:
                composite_img.show()
            except Exception:
                pass

    if not no_json_file:
        json_path = os.path.join(base_dir, output_dir, "predictions.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"samples": samples_collected}, f, ensure_ascii=False, indent=2)
        print(f"Saved structured predictions to {json_path}")

    # Console summary
    for s in samples_collected:
        print("--- Sample", s["sample_index"], "---")
        print("GT:", s["ground_truth"]["name"], "| Pred:", s["prediction"]["name"])
        print(
            "TopK:",
            ", ".join([f"{t['label_name']}({t['prob']:.2f})" for t in s["topk"]]),
        )
        print(
            "Desc:",
            s["description"][:300],
            ("..." if len(s["description"]) > 300 else ""),
        )


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    visualize(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        model_path=args.model_path,
        top_k=args.top_k,
        output_dir=args.output_dir,
        show=not args.no_show,
        composite=args.composite,
        layout=args.layout,
        skip_grid=args.no_grid,
        temperature=args.temperature,
        max_desc_chars=args.max_desc_chars,
        embed_json=args.embed_json,
        json_wrap=args.json_wrap,
        no_json_file=args.no_json_file,
    )


if __name__ == "__main__":
    main()
