import os
import sys
import csv
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
from collections import Counter

# ========== MixUp utils ==========
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0].item()
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def mixup_criterion(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)

def main():
    # ========== Configuration ==========
    ROOT_DIR        = os.path.join(sys.path[0], 'fer2013', 'versions', '1')
    TRAIN_DIR       = os.path.join(ROOT_DIR, 'train')
    VAL_DIR         = os.path.join(ROOT_DIR, 'test')
    CSV_PATH        = os.path.join(sys.path[0], 'V2_val_accuracy.csv')
    BEST_MODEL_PATH = os.path.join(sys.path[0], 'V2_best_model.pth')

    BATCH_SIZE    = 128
    NUM_EPOCHS    = 30
    LEARNING_RATE = 1e-3
    PATIENCE      = 5
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {DEVICE}")

    # ========== Transforms ==========
    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(64),
        transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.15), ratio=(0.3,3.3)),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # ========== Dataset & Sampler ==========
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)

    counts = Counter(label for _, label in train_ds.samples)
    class_weights = {cls: 1.0/count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ========== Model ==========
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feats, 128, bias=False),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, len(train_ds.classes))
    )
    model = model.to(DEVICE)

    # ========== Loss, Optimizer, Scheduler ==========
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE,
        total_steps=total_steps, pct_start=0.3,
        anneal_strategy='cos'
    )
    scaler = amp.GradScaler()

    # ========== Logging & Early Stopping ==========
    best_val_acc = 0.0
    no_improve   = 0
    with open(CSV_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','train_acc','val_loss','val_acc'])

    # ========== Training Loop ==========
    for epoch in range(1, NUM_EPOCHS+1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        model.train()
        t_loss = t_correct = t_total = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS} ▶ Train",
            position=0,
            leave=True
        )
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.4)

            optimizer.zero_grad()
            with amp.autocast():
                outputs = model(imgs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            t_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            t_correct += ((lam * (preds==y_a) + (1-lam) * (preds==y_b)).sum().item())
            t_total += labels.size(0)

            train_bar.set_postfix(
                loss=f"{t_loss/t_total:.4f}",
                acc=f"{100.*t_correct/t_total:.2f}%"
            )

        train_loss, train_acc = t_loss/t_total, 100.*t_correct/t_total

        model.eval()
        v_loss = v_correct = v_total = 0
        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS} ▶ Val",
            position=1,
            leave=True
        )
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                v_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                v_correct += (preds==labels).sum().item()
                v_total += labels.size(0)

                val_bar.set_postfix(
                    loss=f"{v_loss/v_total:.4f}",
                    acc=f"{100.*v_correct/v_total:.2f}%"
                )

        val_loss, val_acc = v_loss/v_total, 100.*v_correct/v_total
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc, no_improve = val_acc, 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 New best: {val_acc:.2f}%")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"⏸ Early stopping after {epoch} epochs")
                break

        with open(CSV_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.4f}", f"{train_acc:.2f}",
                f"{val_loss:.4f}",   f"{val_acc:.2f}"
            ])

    print(f"\nDone — Best Val Acc: {best_val_acc:.2f}% saved to {BEST_MODEL_PATH}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
