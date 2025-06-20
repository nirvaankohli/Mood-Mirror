import os, sys, csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ========== Configuration ==========
ROOT_DIR        = os.path.join(sys.path[0], 'fer2013', 'versions', '1')
TRAIN_DIR       = os.path.join(ROOT_DIR, 'train')
VAL_DIR         = os.path.join(ROOT_DIR, 'test')
CSV_PATH        = os.path.join(sys.path[0], 'V2_validation_accuracy.csv')
BEST_MODEL_PATH = os.path.join(sys.path[0], 'V2_best_model.pth')

BATCH_SIZE    = 128
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-3
PATIENCE      = 5  # quicker early-stop if no gain

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== MixUp Utility ==========
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0].item()
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ========== Data Augmentation & Transforms ==========
train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

val_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ========== Datasets & Loaders ==========
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)

# Compute per-class weights to counter imbalance
counts = torch.tensor([c for _, c in train_ds.class_to_idx.items()])
class_sample_counts = torch.tensor([sum([1 for _, label in train_ds if label==i])
                                    for i in range(len(train_ds.classes))])
class_weights = 1. / (class_sample_counts.float() + 1e-6)
class_weights = class_weights / class_weights.sum() * len(train_ds.classes)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ========== Model: Pretrained ResNet-18 ==========
model = models.resnet18(pretrained=True)
# adapt first conv to accept 1-channel
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# replace classifier
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128, bias=False),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(128, len(train_ds.classes))
)
model = model.to(DEVICE)

# ========== Loss, Optimizer, Scheduler ==========
# criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE),
#                                 label_smoothing=0.1)
# OR for extra hard-sample focus, uncomment FocalLoss:
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = logpt.exp()
        return ((-((1 - pt)**self.gamma) * logpt)).mean()

criterion = FocalLoss(gamma=2.0,
                     weight=class_weights.to(DEVICE))

optimizer = optim.AdamW(model.parameters(),
                        lr=LEARNING_RATE,
                        weight_decay=1e-4)

total_steps = NUM_EPOCHS * len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    total_steps=total_steps,
    pct_start=0.3,
    anneal_strategy='cos'
)

# ========== Logging & Early Stopping ==========
best_val_acc = 0.0
no_improve   = 0

with open(CSV_PATH, 'w', newline='') as f:
    csv.writer(f).writerow([
        'epoch', 'train_loss', 'train_acc',
        'val_loss',  'val_acc'
    ])

# ========== Training Loop ==========
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    t_loss = t_correct = t_total = 0
    train_bar = tqdm(train_loader, desc=f"[{epoch}/{NUM_EPOCHS}] Train")

    for imgs, labels in train_bar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # apply MixUp
        imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=0.4)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        scheduler.step()

        t_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        # approximate acc for mixup (counts only the stronger label)
        t_correct += ((lam * (preds==targets_a) + (1-lam) * (preds==targets_b))
                      .sum().item())
        t_total += labels.size(0)

        train_bar.set_postfix(
            loss=f"{t_loss/t_total:.4f}",
            acc =f"{100.*t_correct/t_total:.2f}%"
        )

    train_loss = t_loss / t_total
    train_acc  = 100. * t_correct / t_total

    # — Validation —
    model.eval()
    v_loss = v_correct = v_total = 0
    val_bar = tqdm(val_loader, desc=f"[{epoch}/{NUM_EPOCHS}] Val  ")

    with torch.no_grad():
        for imgs, labels in val_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            v_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            v_correct += (preds == labels).sum().item()
            v_total += labels.size(0)

            val_bar.set_postfix(
                loss=f"{v_loss/v_total:.4f}",
                acc =f"{100.*v_correct/v_total:.2f}%"
            )

    val_loss = v_loss / v_total
    val_acc  = 100. * v_correct / v_total

    # — Checkpoint & Early Stop —
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        tqdm.write(f"🏆 New best model @ epoch {epoch}: {val_acc:.2f}%")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            tqdm.write(f"⏸ Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # — Log to CSV & console —
    with open(CSV_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_loss:.4f}", f"{train_acc:.2f}",
            f"{val_loss:.4f}",   f"{val_acc:.2f}"
        ])

    tqdm.write(
        f"[{epoch}/{NUM_EPOCHS}] "
        f"Train ▶ loss: {train_loss:.4f}, acc: {train_acc:.2f}% | "
        f" Val ▶ loss: {val_loss:.4f}, acc: {val_acc:.2f}%"
    )

print(f"\nDone. Best Val Acc: {best_val_acc:.2f}% saved to {BEST_MODEL_PATH}")
