import os
import json
import time
import random
import argparse
import multiprocessing
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from Contrastive import Q_cons_fusion, compute_itc_loss
from MLP import MLP_fusion
from process_data import MyData
from config import CONFIG

os.makedirs(CONFIG.checkpoint_dir, exist_ok=True)
os.makedirs(CONFIG.log_dir, exist_ok=True)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CONFIG.seed)

df = pd.read_csv("GLAMI-1M-train.csv")
label_map = {label: idx for idx, label in enumerate(df["category_name"].unique())}
train_dataset = MyData("GLAMI-1M-train.csv", "images", label_map)
test_dataset = MyData("GLAMI-1M-test.csv", "images", label_map)


def args():
    parser = argparse.ArgumentParser(description="Training script for multimodal model")
    parser.add_argument(
        "--model",
        type=str,
        default="Q_cons_fusion",
        help="Model to use: Q_cons_fusion or MLP_fusion",
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    return parser.parse_args()


# train_size = int(0.8 * len(df))

# test_size = len(df) - train_size
# train_dataset, test_dataset = random_split(df, [train_size, test_size])
cpu_ct = multiprocessing.cpu_count()
train_data = DataLoader(
    train_dataset,
    batch_size=CONFIG.batch_size,
    shuffle=True,
    num_workers=min(CONFIG.num_workers, cpu_ct),
    pin_memory=CONFIG.pin_memory,
    drop_last=True,
)
test_data = DataLoader(
    test_dataset,
    batch_size=CONFIG.batch_size,
    shuffle=False,
    num_workers=min(CONFIG.num_workers, cpu_ct),
    pin_memory=CONFIG.pin_memory,
    drop_last=False,
)

print(
    f"CPUs: {cpu_ct} | Train batches: {len(train_data)} | Test batches: {len(test_data)}"
)


def build_scheduler(optimizer):
    if CONFIG.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG.epochs
        )
    if CONFIG.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )
    return None


def train(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    params = (
        model.module.parameters() if hasattr(model, "module") else model.parameters()
    )
    optimizer = AdamW(params, lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    scheduler = build_scheduler(optimizer)
    scaler = GradScaler(enabled=CONFIG.mixed_precision)
    best_acc = 0.0
    best_epoch = 0
    history = []
    no_improve = 0

    for epoch in range(CONFIG.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        train_preds = []
        train_targets = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{CONFIG.epochs}"):
            img = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=CONFIG.mixed_precision):
                output, logit_img, logit_text = model(img, input_ids, attention_mask)
                loss = criterion(output, label) + compute_itc_loss(
                    logit_img, logit_text
                )
            scaler.scale(loss).backward()
            if CONFIG.grad_clip and CONFIG.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            train_preds.extend(output.argmax(1).detach().cpu().tolist())
            train_targets.extend(label.detach().cpu().tolist())

        train_acc = accuracy_score(train_targets, train_preds)
        test_acc = test(model, test_data)

        if scheduler:
            if CONFIG.scheduler == "plateau":
                scheduler.step(1 - test_acc)  # maximize accuracy
            else:
                scheduler.step()

        elapsed = time.time() - t0
        avg_train_loss = total_loss / len(dataloader)
        lr_current = optimizer.param_groups[0]["lr"]
        log_message = (
            f"Epoch {epoch + 1}/{CONFIG.epochs} | Loss {avg_train_loss:.4f} | TrainAcc {train_acc:.4f} | "
            f"TestAcc {test_acc:.4f} | LR {lr_current:.2e} | {elapsed:.1f}s"
        )
        print(log_message)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": avg_train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "lr": lr_current,
                "time_sec": elapsed,
            }
        )

        with open(
            os.path.join(CONFIG.log_dir, "train_log.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(log_message + "\n")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict()
                    if CONFIG.mixed_precision
                    else None,
                    "best_acc": best_acc,
                    "config": CONFIG.__dict__,
                },
                os.path.join(CONFIG.checkpoint_dir, "best_model.pth"),
            )
            print(f"Saved best model (acc {best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= CONFIG.patience:
                print(
                    f"Early stopping at epoch {epoch + 1} (best acc {best_acc:.4f} @ epoch {best_epoch})"
                )
                break

    with open(os.path.join(CONFIG.log_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return best_acc


def test(model, dataloader, check=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            img = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            if check:
                output, _, _ = model(img, input_ids, attention_mask)
            else:
                output = model(img, input_ids, attention_mask)
            preds.extend(output.argmax(1).detach().cpu().tolist())
            targets.extend(label.detach().cpu().tolist())
    return accuracy_score(targets, preds)


def train1(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    scaler = GradScaler(enabled=CONFIG.mixed_precision)
    history = []
    for epoch in range(CONFIG.epochs):
        model.train()
        total_loss = 0.0
        preds = []
        targets = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{CONFIG.epochs}"):
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
            total_loss += loss.item()
            preds.extend(output.argmax(1).detach().cpu().tolist())
            targets.extend(label.detach().cpu().tolist())
        train_acc = accuracy_score(targets, preds)
        test_acc = test(model, test_data, check=False)
        avg_loss = total_loss / len(dataloader)
        log_message = f"Epoch {epoch + 1}/{CONFIG.epochs} | Loss {avg_loss:.4f} | TrainAcc {train_acc:.4f} | TestAcc {test_acc:.4f}"
        print(log_message)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )
        with open(
            os.path.join(CONFIG.log_dir, "attention_log.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(log_message + "\n")
    with open(
        os.path.join(CONFIG.log_dir, "attention_history.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "Q_cons_fusion":
        model = Q_cons_fusion()
        model = torch.nn.DataParallel(model)
        model.to(device)
        train(model, train_data)
    else:
        model = MLP_fusion()
        model = torch.nn.DataParallel(model)
        model.to(device)
        train1(model, train_data)
