import os
import sys
import csv
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

# requires: pip install timm
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

def main():
    # ——— Config ———
    ROOT_DIR        = os.path.join(sys.path[0], 'fer2013', 'versions', '1')
    TRAIN_DIR       = os.path.join(ROOT_DIR, 'train')
    VAL_DIR         = os.path.join(ROOT_DIR, 'test')
    CSV_PATH        = os.path.join(sys.path[0], 'V3_fer2013_val_acc.csv')
    BEST_MODEL_PATH = os.path.join(sys.path[0], 'V3_fer2013_best.pth')

    IMG_SIZE     = 224
    BATCH_SIZE   = 128
    NUM_EPOCHS   = 50
    LR           = 3e-4
    PATIENCE     = 8
    ALPHA        = 0.4   # mixup
    SMOOTHING    = 0.1   # label smoothing
    DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES  = 7

    print(f"➡️ Using device: {DEVICE}")

    # ——— Transforms ———
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3,0.3,0.3,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02,0.15), ratio=(0.3,3.3)),
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # ——— Datasets & Weighted Sampler ———
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)

    counts = Counter(label for _, label in train_ds.samples)
    class_w = {cls: 1.0/count for cls, count in counts.items()}
    samp_w  = [class_w[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(samp_w, num_samples=len(samp_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ——— Model ———
    model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # ——— Mixup/CutMix + Loss ———
    mixup_fn = Mixup(
        mixup_alpha=ALPHA,
        cutmix_alpha=1.0,
        prob=0.8,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=SMOOTHING,
    )
    criterion = SoftTargetCrossEntropy()

    # ——— Optimizer, Scheduler, AMP ———
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = amp.GradScaler()

    # ——— Logging & Early Stop ———
    best_acc   = 0.0
    no_improve = 0
    with open(CSV_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','train_acc','val_loss','val_acc'])

    for epoch in range(1, NUM_EPOCHS+1):
        print(f"\n🔄 Epoch {epoch}/{NUM_EPOCHS}")
        # — Train —
        model.train()
        t_loss = t_correct = t_total = 0
        pbar = tqdm(train_loader, desc="Train", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs, labels = mixup_fn(imgs, labels)

            optimizer.zero_grad()
            with amp.autocast():
                preds = model(imgs)
                loss  = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # track training stats (approximate)
            t_loss  += loss.item() * imgs.size(0)
            preds_h = preds.argmax(dim=1)
            targets_h = labels.argmax(dim=1)
            t_correct += (preds_h == targets_h).sum().item()
            t_total   += labels.size(0)

            pbar.set_postfix(loss=f"{t_loss/t_total:.4f}",
                             acc =f"{100.*t_correct/t_total:.2f}%")

        train_loss = t_loss / t_total
        train_acc  = 100. * t_correct / t_total

        # — Validate —
        model.eval()
        v_loss = v_correct = v_total = 0
        vbar = tqdm(val_loader, desc=" Val", leave=False)
        with torch.no_grad():
            for imgs, labels in vbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                loss   = criterion(logits, labels)

                v_loss   += loss.item() * imgs.size(0)
                preds    = logits.argmax(dim=1)
                v_correct+= (preds == labels).sum().item()
                v_total  += labels.size(0)

                vbar.set_postfix(loss=f"{v_loss/v_total:.4f}",
                                 acc =f"{100.*v_correct/v_total:.2f}%")

        val_loss = v_loss / v_total
        val_acc  = 100. * v_correct / v_total

        # — Checkpoint & Early Stop —
        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 New best: {val_acc:.2f}%")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"⏸ Stopping early after {epoch} epochs")
                break

        # — Log —
        with open(CSV_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.4f}", f"{train_acc:.2f}",
                f"{val_loss:.4f}",   f"{val_acc:.2f}"
            ])

        # — LR Scheduler step (per epoch) —
        scheduler.step()

    print(f"\n✅ Done! Best Val Acc: {best_acc:.2f}% (→{BEST_MODEL_PATH})")

if __name__ == "__main__":
    main()
