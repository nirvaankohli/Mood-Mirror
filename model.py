import os
import sys
import csv
from collections import Counter

import torch
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from timm import create_model
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import CosineLRScheduler
import numpy as np
from tqdm import tqdm


# ——— Helpers for MixUp & CutMix ———
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    return bbx1, bby1, bbx2, bby2

def mixup_cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    """Randomly apply MixUp or CutMix."""
    if alpha <= 0:
        return x, y, y, 1.0, 'none'
    lam = np.random.beta(alpha, alpha)
    if np.random.rand() < cutmix_prob:
        # CutMix
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[torch.randperm(x.size(0)), :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        y_a, y_b = y, y[torch.randperm(x.size(0))]
        return x, y_a, y_b, lam, 'cutmix'
    else:
        # MixUp
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        y_a, y_b = y, y[idx]
        return x, y_a, y_b, lam, 'mixup'


def main():
    # ——— Config ———
    
    ROOT_DIR        = os.path.join(sys.path[0], 'fer2013', 'versions', '1')
    TRAIN_DIR       = os.path.join(ROOT_DIR, 'train')
    VAL_DIR         = os.path.join(ROOT_DIR, 'test')
    CSV_PATH        = os.path.join(sys.path[0], 'V4_validation_accuracy.csv')
    BEST_MODEL_PATH = os.path.join(sys.path[0], 'V4_best_model.pth')

    BATCH_SIZE    = 64
    NUM_EPOCHS    = 75
    LR            = 1e-3
    PATIENCE      = 8
    ALPHA         = 0.4   # for MixUp/CutMix
    SMOOTHING     = 0.1   # label smoothing
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES   = 7
    IMG_SIZE      = 112   # increase from 64 → 112

    # ——— Transforms / Augmentations ———
    
    train_tf = transforms.Compose([

        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ])

    val_tf = transforms.Compose([
        
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ])

    # ——— Datasets & Sampler ———
    
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)

    counts = Counter(label for _, label in train_ds.samples)
    class_w = {cls: 1.0 / count for cls, count in counts.items()}
    samp_w  = [class_w[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)

    train_loader = DataLoader(

        train_ds, batch_size=BATCH_SIZE,

        sampler=sampler, num_workers=4, pin_memory=True

    )
    val_loader = DataLoader(

        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True

    )

    # ——— Model ———

    model = create_model(

        'tf_efficientnetv2_s.in21k',
        pretrained=True,
        num_classes=NUM_CLASSES

    ).to(DEVICE)

    # ——— Loss, Optimizer, Scheduler, AMP ———

    criterion = LabelSmoothingCrossEntropy(smoothing=SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineLRScheduler(

        optimizer,
        t_initial=NUM_EPOCHS * len(train_loader),
        lr_min=1e-6,
        warmup_lr_init=1e-7,
        warmup_t=5 * len(train_loader),
        cycle_limit=1,
        t_in_epochs=False

    )

    scaler = amp.GradScaler()

    # ——— Logging & Early Stop ———

    best_acc, no_improve = 0.0, 0

    with open(CSV_PATH, 'w', newline='') as f:

        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    # For per-iteration LR scheduling:

    global_step = 0

    for epoch in range(1, NUM_EPOCHS + 1):

        print(f"\n🔄 Epoch {epoch}/{NUM_EPOCHS}")
        
        # — Train —

        model.train()
        t_loss = t_correct = t_total = 0
        pbar = tqdm(train_loader, desc="Train", leave=False)

        for imgs, labels in pbar:
            
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs, y_a, y_b, lam, aug_type = mixup_cutmix_data(imgs, labels, ALPHA)

            optimizer.zero_grad()
            
            with amp.autocast():
                preds = model(imgs)
                loss = lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # step the scheduler per iteration
            scheduler.step_update(global_step)
            global_step += 1

            t_loss += loss.item() * imgs.size(0)
            top1 = preds.argmax(1)
            
            correct = (lam * (top1 == y_a) + (1 - lam) * (top1 == y_b)).sum().item()
            t_correct += correct
            t_total += labels.size(0)
            pbar.set_postfix(
                loss=f"{t_loss / t_total:.4f}",
                acc=f"{100. * t_correct / t_total:.2f}%"
            )

        train_loss = t_loss / t_total
        train_acc = 100. * t_correct / t_total

        # — Validate —
        model.eval()
        v_loss = v_correct = v_total = 0
        vbar = tqdm(val_loader, desc=" Val", leave=False)
        with torch.no_grad():
            for imgs, labels in vbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, labels)

                v_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
                vbar.set_postfix(
                    loss=f"{v_loss / v_total:.4f}",
                    acc=f"{100. * v_correct / v_total:.2f}%"
                )

        val_loss = v_loss / v_total
        val_acc = 100. * v_correct / v_total

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
                f"{train_loss:.4f}",
                f"{train_acc:.2f}",
                f"{val_loss:.4f}",
                f"{val_acc:.2f}"
            ])

    print(f"\n✅ Done! Best Val Acc: {best_acc:.2f}% (→{BEST_MODEL_PATH})")


if __name__ == '__main__':
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except ImportError:
        pass
    main()
