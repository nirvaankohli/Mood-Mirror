import os
import sys
import csv
import random
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():

    p = argparse.ArgumentParser(description="FER2013 EfficientNet Training")
    p.add_argument(
        "--data-root",
        type=str,
        default=os.path.join(sys.path[0], "fer2013", "versions", "1"),
    )

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.4, help="MixUp α")
    p.add_argument("--smooth", type=float, default=0.1, help="Label smoothing")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    # Put all logs under sys.path[0]/logs/fer2013_TIMESTAMP
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join(sys.path[0], "logs", f"fer2013_{timestamp}"),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint in log-dir",
    )
    return p.parse_args()


def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def make_dataloaders(root, batch_size, num_workers):

    train_tf = transforms.Compose([

        transforms.Grayscale(3),
        transforms.RandomResizedCrop(64, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.15), ratio=(0.3,3.3)),
    
    ])

    val_tf = transforms.Compose([

        transforms.Grayscale(3),
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    
    ])

    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(root, "test"),  transform=val_tf)

    counts = Counter(label for _, label in train_ds.samples)
    class_weights = {cls: 1.0/count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, 
                              batch_size=batch_size,
                              sampler=sampler, 
                              num_workers=num_workers, 
                              pin_memory=True)
    
    val_loader   = DataLoader(val_ds,   
                              batch_size=batch_size,
                              shuffle=False,      
                              num_workers=num_workers, 
                              pin_memory=True)
    
    return train_loader, val_loader


def build_model(num_classes, device):


    model = models.efficientnet_b0(pretrained=True)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, alpha, clip_grad=None):
    
    model.train()


    running_loss = running_correct = running_total = 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):

        imgs, labels = imgs.to(device), labels.to(device)
        mixed_x, y_a, y_b, lam = mixup_data(imgs, labels, alpha)

        optimizer.zero_grad()

        with amp.autocast():

            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        scaler.scale(loss).backward()

        if clip_grad:

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss   += loss.item() * imgs.size(0)
        preds           = outputs.argmax(1)
        running_correct += (lam * (preds==y_a) + (1-lam)*(preds==y_b)).sum().item()
        running_total   += labels.size(0)

    return running_loss/running_total, 100.*running_correct/running_total


def validate(model, loader, criterion, device):

    model.eval()
    
    val_loss = val_correct = val_total = 0

    with torch.no_grad():

        for imgs, labels in tqdm(loader, desc="Val  ", leave=False):

            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            val_loss    += loss.item() * imgs.size(0)
            preds        = logits.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    return val_loss/val_total, 100.*val_correct/val_total


def main():

    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_dataloaders(args.data_root, args.batch_size, args.num_workers)
    model     = build_model(num_classes=7, device=DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smooth)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps = args.epochs * len(train_loader)
    scheduler   = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=args.lr,
                                                total_steps=total_steps,
                                                pct_start=0.3, 
                                                anneal_strategy="cos")
    
    scaler = amp.GradScaler()

    # TensorBoard & CSV log (both under sys.path[0])
    writer   = SummaryWriter(log_dir=args.log_dir)
    csv_path = os.path.join(args.log_dir, "metrics.csv")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr"])

    # Resume?
    start_epoch, best_acc = 1, 0.0
    ckpt_path = os.path.join(args.log_dir, "best.pth")

    if args.resume and os.path.exists(ckpt_path):

        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])

        start_epoch = state["epoch"] + 1
        best_acc    = state["best_acc"]

    no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):

        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        tloss, tacc = train_one_epoch(model, 
                                      train_loader,
                                      optimizer, 
                                      criterion,
                                      scaler, 
                                      scheduler,
                                      DEVICE, args.alpha,
                                      clip_grad=1.0)
        
        vloss, vacc = validate(model, 
                               val_loader, 
                               criterion, 
                               DEVICE)

        lr_now = scheduler.get_last_lr()[0]

        writer.add_scalars("Loss", {"train": tloss, "val": vloss}, epoch)
        writer.add_scalars("Acc",  {"train": tacc,  "val": vacc}, epoch)
        writer.add_scalar("LR", lr_now, epoch)

        with open(csv_path, "a", newline="") as f:

            csv.writer(f).writerow([epoch, f"{tloss:.4f}", f"{tacc:.2f}",
                                     f"{vloss:.4f}", f"{vacc:.2f}", f"{lr_now:.6f}"])

        if vacc > best_acc:
            
            best_acc, no_improve = vacc, 0

            torch.save({

                "model": model.state_dict(),
                "opt":   optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc

            }, ckpt_path)

            print(f"🏆 New best val acc: {best_acc:.2f}%")

        else:

            no_improve += 1

            if no_improve >= args.patience:

                print(f"⏸ Early stopping at epoch {epoch}")

                break

    writer.close()

    print(f"\n✅ Done! Best Val Acc: {best_acc:.2f}% → {ckpt_path}")


if __name__ == "__main__":
    main()
