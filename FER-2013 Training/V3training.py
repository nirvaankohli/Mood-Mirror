import os
import sys
import csv
from collections import Counter
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ——— MixUp + CutMix utility ———
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = torch.sqrt(1. - lam)
    cut_w = (W * cut_rat).type(torch.long)
    cut_h = (H * cut_rat).type(torch.long)

    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)

    return bbx1, bby1, bbx2, bby2

def mixup_cutmix(x, y, alpha_mix=0.4, alpha_cut=1.0, p_cutmix=0.5):
    """Randomly apply MixUp or CutMix per batch."""
    if alpha_mix > 0 and (random.random() > p_cutmix):
        # MixUp
        lam = torch._sample_dirichlet(torch.tensor([alpha_mix, alpha_mix]))[0].item()
        idx = torch.randperm(x.size(0), device=x.device)
        x2, y2 = x[idx], y[idx]
        mixed_x = lam * x + (1 - lam) * x2
        return mixed_x, y, y2, lam
    else:
        # CutMix
        lam = torch._sample_dirichlet(torch.tensor([alpha_cut, alpha_cut]))[0].item()
        idx = torch.randperm(x.size(0), device=x.device)
        x2, y2 = x[idx], y[idx]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
        # adjust lambda to exact pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return mixed_x, y, y2, lam

def criterion_mix(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)

def main():
    # ——— Config ———
    ROOT_DIR        = os.path.join(sys.path[0], 'fer2013', 'versions', '1')
    TRAIN_DIR       = os.path.join(ROOT_DIR, 'train')
    VAL_DIR         = os.path.join(ROOT_DIR, 'test')
    CSV_PATH        = os.path.join(sys.path[0], 'V3_fer2013_val_acc.csv')
    BEST_MODEL_PATH = os.path.join(sys.path[0], 'V3_fer2013_best.pth')

    BATCH_SIZE      = 64    # larger models + bigger images → smaller batches
    NUM_EPOCHS      = 80
    LR              = 2e-4  # lower lr for bigger nets
    PATIENCE        = 10
    ALPHA_MIX       = 0.4
    ALPHA_CUT       = 1.0
    P_CUTMIX        = 0.5
    SMOOTHING       = 0.1
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES     = 7
    UNFREEZE_EPOCH  = 5    # freeze backbone first few epochs

    print(f"➡️ Using device: {DEVICE}")

    # ——— Transforms ———
    IMG_SIZE = 224
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    # ——— Datasets & Weighted Sampler ———
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)

    counts = Counter(label for _,label in train_ds.samples)
    class_w = {cls: 1.0/count for cls,count in counts.items()}
    samp_w = [class_w[label] for _,label in train_ds.samples]
    sampler = WeightedRandomSampler(samp_w, num_samples=len(samp_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ——— Model ———
    model = models.efficientnet_b3(pretrained=True)  # bigger than B0
    # freeze all except classifier head for first few epochs
    for param in model.features.parameters():
        param.requires_grad = False

    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, NUM_CLASSES)
    model = model.to(DEVICE)

    # ——— Loss, Optimizer, Scheduler, AMP ———
    criterion = nn.CrossEntropyLoss(label_smoothing=SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR*5, total_steps=total_steps,
        pct_start=0.2, anneal_strategy='cos'
    )
    scaler = amp.GradScaler()

    # ——— Logging & Early Stop ———
    best_acc = 0.0
    no_improve = 0
    with open(CSV_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','train_acc','val_loss','val_acc'])

    for epoch in range(1, NUM_EPOCHS+1):
        print(f"\n🔄 Epoch {epoch}/{NUM_EPOCHS}")

        # — Unfreeze backbone? —
        if epoch == UNFREEZE_EPOCH:
            print("🔓 Unfreezing backbone for fine‐tuning")
            for param in model.features.parameters():
                param.requires_grad = True

        # — Train —
        model.train()
        t_loss = t_correct = t_total = 0
        pbar = tqdm(train_loader, desc=" Train", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs, y_a, y_b, lam = mixup_cutmix(imgs, labels, ALPHA_MIX, ALPHA_CUT, P_CUTMIX)

            optimizer.zero_grad()
            with amp.autocast():
                preds = model(imgs)
                loss  = criterion_mix(criterion, preds, y_a, y_b, lam)
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

        # — Validate with TTA ——
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="  Val", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                # simple 2‐crop TTA
                crops = torch.stack([
                    transforms.functional.resize(imgs, (IMG_SIZE,IMG_SIZE)),
                    transforms.functional.center_crop(imgs, IMG_SIZE)
                ], dim=1)  # shape [B,2,C,H,W]
                bs, ncrops, C, H, W = crops.shape
                flat = crops.view(-1, C, H, W)
                logits = model(flat)
                logits = logits.view(bs, ncrops, -1).mean(1)

                loss = criterion(logits, labels)
                v_loss   += loss.item() * bs
                preds     = logits.argmax(1)
                v_correct+= (preds==labels).sum().item()
                v_total  += bs

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

    print(f"\n✅ Done! Best Val Acc: {best_acc:.2f}% → {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()
