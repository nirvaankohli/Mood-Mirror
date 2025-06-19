import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import csv

# ========== Configuration ==========
ROOT_DIR        = os.path.join(sys.path[0], 'fer2013', 'versions', '1')
TRAIN_DIR       = os.path.join(ROOT_DIR, 'train')
VAL_DIR         = os.path.join(ROOT_DIR, 'test')
CSV_PATH        = os.path.join(sys.path[0], 'validation_accuracy.csv')
BEST_MODEL_PATH = os.path.join(sys.path[0], 'best_model.pth')

BATCH_SIZE    = 64
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3
DEVICE        = torch.device('cpu')

# Early stopping if no improvement in N epochs
PATIENCE = 7

# ========== Data Augmentation & Transforms ==========
train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.RandomResizedCrop(48, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])
val_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ========== DataLoaders ==========
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ========== Custom Blocks & Model ==========
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w

class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch, reduction=8)
        self.relu  = nn.ReLU(inplace=True)
        self.down  = (nn.Sequential(
                          nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(out_ch)
                      ) if downsample else nn.Identity())
        self.drop  = nn.Dropout2d(0.1)

    def forward(self, x):
        identity = self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        out = self.se(out)
        out += identity
        return self.relu(out)

class CustomResNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Stages with increasing channels
        self.stage1 = self._make_stage(32,  64,  blocks=2, downsample=True)
        self.stage2 = self._make_stage(64, 128, blocks=2, downsample=True)
        self.stage3 = self._make_stage(128,256, blocks=2, downsample=True)
        self.dropout = nn.Dropout(0.5)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _make_stage(self, 
                    in_ch, 
                    out_ch, blocks, downsample):
        layers = [ResidualSEBlock(in_ch, out_ch, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualSEBlock(out_ch, out_ch, downsample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.dropout(x)
        x = self.pool(x)
        return self.classifier(x)

model = CustomResNet().to(DEVICE)

# ========== Loss, Optimizer, Scheduler ==========
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                 mode='max',
                                                 factor=0.5, 
                                                 patience=3)

# ========== Logging & Early Stopping Setup ==========
best_val_acc = 0.0
no_improve   = 0
with open(CSV_PATH, 'w', newline='') as f:
    csv.writer(f).writerow([
        'epoch',
        'train_loss', 'train_acc',
        'val_loss',   'val_acc'
    ])

# ========== Training Loop ==========
for epoch in range(1, NUM_EPOCHS + 1):
    # — Training —
    model.train()
    train_bar = tqdm(train_loader,
                     desc=f"[{epoch}/{NUM_EPOCHS}] Train",
                     position=0, leave=True)
    t_loss = t_correct = t_total = 0
    for imgs, labels in train_bar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        t_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        t_correct += (preds == labels).sum().item()
        t_total += labels.size(0)
        train_bar.set_postfix(
            loss=f"{t_loss/t_total:.4f}",
            acc=f"{100.*t_correct/t_total:.2f}%"
        )
    train_loss = t_loss / t_total
    train_acc  = 100. * t_correct / t_total

    # — Validation —
    model.eval()
    val_bar = tqdm(val_loader,
                   desc=f"[{epoch}/{NUM_EPOCHS}] Val  ",
                   position=1, leave=True)
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for imgs, labels in val_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)

            v_loss += loss.item() * imgs.size(0)
            preds = out.argmax(1)
            v_correct += (preds == labels).sum().item()
            v_total += labels.size(0)
            val_bar.set_postfix(
                loss=f"{v_loss/v_total:.4f}",
                acc=f"{100.*v_correct/v_total:.2f}%"
            )
    val_loss = v_loss / v_total
    val_acc  = 100. * v_correct / v_total
    scheduler.step(val_acc)

    # — Checkpoint & early stop —
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve   = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        tqdm.write(f"🏆 New best model @ epoch {epoch} — Val Acc: {val_acc:.2f}%")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            tqdm.write(f"⏸ Early stopping: no improvement for {PATIENCE} epochs.")
            break

    # — Log to CSV & console summary —
    with open(CSV_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_loss:.4f}", f"{train_acc:.2f}",
            f"{val_loss:.4f}",   f"{val_acc:.2f}"
        ])
    tqdm.write(
        f"[{epoch}/{NUM_EPOCHS}] "
        f"Train ▶ loss: {train_loss:.4f}, acc: {train_acc:.2f}% | "
        f" Val ▶ loss: {val_loss:.4f}, acc: {val_acc:.2f}% | "
        f"LR: {optimizer.param_groups[0]['lr']:.1e}"
    )

print(f"\nDone. Best Val Acc: {best_val_acc:.2f}% — model saved to {BEST_MODEL_PATH}")
