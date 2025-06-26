import os
import sys
import csv
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ——— MixUp utility ———
def mixup_data(x, y, alpha=0.4):

    if alpha > 0:

        lam = torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0].item()
    else:
        lam = 1.0

    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1. - lam) * x[idx]
    y_a, y_b = y, y[idx]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(crit, pred, y_a, y_b, lam):

    return lam * crit(pred, y_a) + (1. - lam) * crit(pred, y_b)


def main():

    # ——— Config ———

    ROOT_DIR        = os.path.join(sys.path[0], 'fer2013', 'versions', '1')
    TRAIN_DIR       = os.path.join(ROOT_DIR, 'train')
    VAL_DIR         = os.path.join(ROOT_DIR, 'test')
    CSV_PATH        = os.path.join(sys.path[0], 'V2_fer2013_val_acc.csv')
    BEST_MODEL_PATH = os.path.join(sys.path[0], 'V2_fer2013_best.pth')

    BATCH_SIZE    = 64
    NUM_EPOCHS    = 50
    LR            = 1e-3
    PATIENCE      = 8
    ALPHA         = 0.4   # mixup
    SMOOTHING     = 0.1   # label smoothing
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES   = 7

    print(f"➡️ Using device: {DEVICE}")

    # ——— Transforms ———

    train_tf = transforms.Compose([

        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(64, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.15), ratio=(0.3,3.3)),

    ])
    val_tf = transforms.Compose([

        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # ——— Datasets & Weighted Sampler ———
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)

    # balance classes by inverse frequency
    3
    counts = Counter(label for _,label in train_ds.samples)
    class_w = {cls: 1.0/count for cls,count in counts.items()}
    samp_w  = [class_w[label] for _,label in train_ds.samples]
    sampler = WeightedRandomSampler(samp_w, num_samples=len(samp_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ——— Model ———
    model = models.efficientnet_b0(pretrained=True)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, NUM_CLASSES)
    model = model.to(DEVICE)

    # ——— Loss, Opt, Scheduler, AMP ———
    criterion = nn.CrossEntropyLoss(label_smoothing=SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
        pct_start=0.3, anneal_strategy='cos'
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
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, ALPHA)

            optimizer.zero_grad()
            with amp.autocast():
                preds = model(imgs)
                loss  = mixup_criterion(criterion, preds, y_a, y_b, lam)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            t_loss   += loss.item() * imgs.size(0)
            top1     = preds.argmax(1)
            correct  = (lam * (top1==y_a) + (1.-lam)*(top1==y_b)).sum().item()
            t_correct+= correct
            t_total  += labels.size(0)

            pbar.set_postfix(loss=f"{t_loss/t_total:.4f}",
                             acc =f"{100.*t_correct/t_total:.2f}%")

        train_loss = t_loss / t_total
        train_acc  = 100.*t_correct / t_total

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
                preds     = logits.argmax(1)
                v_correct+= (preds==labels).sum().item()
                v_total  += labels.size(0)

                vbar.set_postfix(loss=f"{v_loss/v_total:.4f}",
                                 acc =f"{100.*v_correct/v_total:.2f}%")

        val_loss = v_loss / v_total
        val_acc  = 100.*v_correct / v_total

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
            csv.writer(f).writerow([epoch,
                                    f"{train_loss:.4f}", f"{train_acc:.2f}",
                                    f"{val_loss:.4f}",   f"{val_acc:.2f}"])

    print(f"\n✅ Done! Best Val Acc: {best_acc:.2f}% (→{BEST_MODEL_PATH})")

if __name__ == "__main__":
    main()
