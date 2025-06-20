
import os
import sys
import argparse
import csv
from collections import Counter
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models

# ─── Spatial Attention ─────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn

# ─── FER Model ────────────────────────────────────────────────────────────────
class FERModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Load pretrained VGG16-BN
        vgg = models.vgg16_bn(pretrained=True)
        # Replace first conv to accept 1-channel input
        orig = vgg.features[0]
        new0 = nn.Conv2d(1, orig.out_channels,
                         kernel_size=orig.kernel_size,
                         stride=orig.stride,
                         padding=orig.padding,
                         bias=(orig.bias is not None))
        # Sum weights over original 3 channels
        new0.weight.data = orig.weight.data.sum(dim=1, keepdim=True)
        vgg.features[0] = new0

        # Freeze early layers, fine-tune from layer4 onward
        for name, param in vgg.features.named_parameters():
            layer_idx = int(name.split('.')[0])
            param.requires_grad = (layer_idx >= 24)

        self.features = vgg.features
        self.attn     = SpatialAttention(kernel_size=7)
        self.pool     = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
            # raw logits here! Softmax is in loss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.pool(x)
        return self.classifier(x)

# ─── Focal Loss (optional) ────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = -self.ce(inputs, targets)
        p = logp.exp()
        loss = -((1 - p) ** self.gamma) * logp
        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
        else:
            return loss

# ─── Training / Validation loops ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, clip_grad):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = running_corrects / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = running_corrects / total
    return epoch_loss, epoch_acc

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train FER-2013 with improved script")
    parser.add_argument('--data-dir', type=str, required=True,
                        help="Root folder with `train/` and `test/` subdirs")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--early-stop-patience', type=int, default=7)
    parser.add_argument('--use-focal-loss', action='store_true',
                        help="Use Focal Loss instead of standard CrossEntropyLoss")
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help="Label smoothing factor (0 means no smoothing)")
    parser.add_argument('--mixed-precision', action='store_true',
                        help="Enable mixed-precision training")
    
    parser.add_argument('--grad-clip', type=float, default=2.0)
    parser.add_argument('--log-dir', type=str, default='logs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # ── Transforms ────────────────────────────────────────────────────────────────
    
    train_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48,48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0,
                                translate=(0.1,0.1),
                                scale=(0.9,1.1)),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.15), ratio=(0.3,3.3))
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ── Datasets & Sampler ───────────────────────────────────────────────────────
    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                    transform=train_tf)
    
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, 'test'),
                                    transform=val_tf)

    # compute class weights
    counts = Counter([y for _, y in train_ds.samples])
    cls_weights = [1.0 / counts[i] for i in range(len(counts))]
    sample_weights = [cls_weights[y] for _, y in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # ── Model, Loss, Optimizer, Scheduler ───────────────────────────────────────
    model = FERModel(num_classes=len(counts)).to(device)

    if args.use_focal_loss:
        loss_fn = FocalLoss(gamma=2.0,
                            weight=torch.tensor(cls_weights).to(device))
    else:
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(cls_weights).to(device),
            label_smoothing=args.label_smoothing
        )

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # ── CSV logging setup ────────────────────────────────────────────────────────
    csv_path = os.path.join(args.log_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['epoch',
                             'train_loss','train_acc',
                             'val_loss','val_acc'])
    best_val_acc = 0.0
    epochs_no_improve = 0

    # ── Training Loop ────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer,
            device, scaler, args.grad_clip
        )
        val_loss,   val_acc   = validate_one_epoch(
            model, val_loader, loss_fn, device
        )

        scheduler.step()

        # TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        # CSV
        with open(csv_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch,
                                 f"{train_loss:.4f}", f"{train_acc:.4f}",
                                 f"{val_loss:.4f}",   f"{val_acc:.4f}"])

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))
            print(f"🏆 Epoch {epoch}: new best val_acc={val_acc:.4f}")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.early_stop_patience:
            print(f"⏸️ Early stopping at epoch {epoch}")
            break

        print(f"Epoch {epoch}/{args.epochs} "
              f"- train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
              f"- val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

    print(f"Training complete — best val_acc = {best_val_acc:.4f}")
    writer.close()

if __name__ == "__main__":
    freeze_support()
    main()
