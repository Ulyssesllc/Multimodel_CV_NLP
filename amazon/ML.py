import os
import time
import math
import json
import random
import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import accuracy_score
from process_data import MyData
from MLP import FusionModule
from config import CONFIG
from torch.cuda.amp import autocast, GradScaler

base_dir = os.path.dirname(__file__)
csv_file = os.path.join(base_dir, "training_data.csv")
# Attempt to find image directory under amazon/, else fallback to root folder
candidate = os.path.join(base_dir, "amazon_dataset")
if os.path.isdir(candidate):
    img_dir = candidate
else:
    img_dir = os.path.join(os.path.dirname(base_dir), "amazon_dataset")
dataset = MyData(csv_file, img_dir)
num_classes = len(dataset.label_map)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CONFIG.seed)

train_data = DataLoader(
    train_dataset,
    batch_size=CONFIG.batch_size,
    shuffle=True,
    num_workers=CONFIG.num_workers,
    pin_memory=CONFIG.pin_memory,
)
test_data = DataLoader(
    test_dataset,
    batch_size=CONFIG.batch_size,
    shuffle=False,
    num_workers=CONFIG.num_workers,
    pin_memory=CONFIG.pin_memory,
)

model = FusionModule(num_classes=num_classes, dropout=CONFIG.dropout)


def build_scheduler(optimizer):
    if CONFIG.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.epochs)
    if CONFIG.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    return None


def train(model, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG.lr,
        weight_decay=CONFIG.weight_decay,
    )
    scheduler = build_scheduler(optimizer)
    scaler = GradScaler(enabled=CONFIG.mixed_precision)

    start_epoch = 0
    best_loss = math.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve = 0

    if CONFIG.resume_checkpoint and os.path.isfile(CONFIG.resume_checkpoint):
        ckpt = torch.load(CONFIG.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt and CONFIG.mixed_precision:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", best_loss)
        print(f"Resumed from {CONFIG.resume_checkpoint} at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, CONFIG.epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for batch in train_loader:
            img = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=CONFIG.mixed_precision):
                output = model(img, input_ids, attention_mask)
                loss = criterion(output, label)
            scaler.scale(loss).backward()
            if CONFIG.grad_clip and CONFIG.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            train_preds.extend(output.argmax(1).detach().cpu().tolist())
            train_targets.extend(label.detach().cpu().tolist())

        avg_train_loss = sum(train_losses) / len(train_losses)
        train_acc = accuracy_score(train_targets, train_preds)

        # validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label = batch["label"].to(device)
                with autocast(enabled=CONFIG.mixed_precision):
                    output = model(img, input_ids, attention_mask)
                    loss = criterion(output, label)
                val_losses.append(loss.item())
                val_preds.extend(output.argmax(1).detach().cpu().tolist())
                val_targets.extend(label.detach().cpu().tolist())

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        val_acc = accuracy_score(val_targets, val_preds) if val_preds else 0.0

        if scheduler:
            if CONFIG.scheduler == "plateau":
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{CONFIG.epochs} | Train Loss {avg_train_loss:.4f} Acc {train_acc:.3f} | "
            f"Val Loss {avg_val_loss:.4f} Acc {val_acc:.3f} | LR {lr_current:.2e} | {elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": lr_current,
                "time_sec": elapsed,
            }
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict()
                    if CONFIG.mixed_precision
                    else None,
                    "best_loss": best_loss,
                },
                "checkpoint_best.pth",
            )
        else:
            no_improve += 1
            if no_improve >= CONFIG.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_wts)
    with open("training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return model


if __name__ == "__main__":
    trained_model = train(model, train_data, test_data)
    torch.save(trained_model.state_dict(), "fusion_model_best.pth")
    print("Saved best model to fusion_model_best.pth")
