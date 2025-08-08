import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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


def main(num_samples=5, batch_size=1, model_path="fusion_model_best.pth"):
    # Setup dataset
    base_dir = os.path.dirname(__file__)
    csv_file = os.path.join(base_dir, "training_data.csv")
    img_dir = os.path.join(base_dir, "amazon_dataset")
    dataset = MyData(csv_file, img_dir)
    # Create inverse label map for display
    label_map_inv = {v: k for k, v in dataset.label_map.items()}

    # DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model checkpoint and infer number of classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    num_classes_checkpoint = state_dict["final_fc.weight"].shape[0]
    model = FusionModule(num_classes=num_classes_checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Plot samples
    num_cols = num_samples
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        img = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].item()
        # get original description text
        desc = batch.get("description")
        if isinstance(desc, (list, tuple)):
            desc = desc[0]
        # truncate description if too long
        desc_display = (desc[:50] + "...") if len(desc) > 50 else desc

        with torch.no_grad():
            output = model(img, input_ids, attention_mask)
            pred = torch.argmax(output, dim=1).item()

        ax = axes[i] if num_cols > 1 else axes
        imshow(batch["image"].squeeze(0), ax)
        # display description, ground-truth and prediction
        gt_name = label_map_inv.get(label, str(label))
        pred_name = label_map_inv.get(pred, str(pred))
        full_title = f"Desc: {desc_display}\nGT: {gt_name} | Pred: {pred_name}"
        ax.set_title(full_title, fontsize=8)

    plt.tight_layout()
    # Save visualization to file
    save_path = os.path.join(base_dir, "predictions.png")
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
