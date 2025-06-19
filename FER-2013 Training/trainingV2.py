import os
import numpy as np
from collections import Counter

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

# -=-=-=-=-=-=-=-= Configuration =-=-=-=-=-=-=-

DATA_DIR = os.path.join(sys.path[0], 'fer2013', 'versions', '1')
BATCH_SIZE = 64
NUM_CLASSES = 7
NUM_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -==-=-=-=-=-=- Data Aug & Transforms -=-=-=-=-=-

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
    ),
    transforms.ColorJitter(brightness=0.2, 
                           contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,), 
        (0.5,)),
    transforms.RandomErasing(
        p=0.5, 
        scale=(0.02, 0.15), 
        ratio=(0.3, 3.3)
    ),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,), 
        (0.5,)),
])

# -=-=-=-=-=-=- DataLoaders & Samplers -=-=-=-=-=-

train_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, 'train'),
    transform=train_transform)

val_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, 'test'),
    transform=val_transform)

# Calculate class weights for balanced sampling

targets = [s[1] for s in train_ds.samples]

class_counts = Counter(targets)
class_counts = Counter(targets)
class_weights = [0]*NUM_CLASSES

for cls, cnt in class_counts.items():
    class_weights[cls] = 1.0 / cnt

sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, 
                                num_samples=len(sample_weights), 
                                replacement=True)

train_loader = DataLoader(train_ds, 
                          batch_size=BATCH_SIZE, 
                          sampler=sampler, 
                          num_workers=4)

val_loader   = DataLoader(val_ds,   
                          batch_size=BATCH_SIZE, 
                          shuffle=False,      
                          num_workers=4)

class_weights_tensor = torch.tensor(class_weights, 
                                    dtype=torch.float).to(DEVICE)

# -=-=-=-=-=- Spacial Attention Module -=-=-=-=-=-=-

class SpatialAttention(nn.Module):
    
    def __init__(self, kernel_size=7):

        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)

    def forward(self, x):

        avg_out = torch.mean(x, 
                             dim=1, 
                             keepdim=True)
        max_out, _ = torch.max(x, 
                               dim=1, 
                               keepdim=True)
        
        x = torch.cat([avg_out, max_out], 
                      dim=1)
        
        x = self.conv(x)
        return torch.sigmoid(x) * x
    
# -=-=-=-=-=- FER Model with Pretrained VGG Backbone -=-=-=-=-=-

class FERModel(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        vgg = models.vgg16_bn(pretrained=True)

        orig = vgg.features[0]
        new0 = nn.Conv2d(1, orig.out_channels, #FER-2013 uses grayscale images
                         kernel_size=orig.kernel_size,
                         stride=orig.stride,
                         padding=orig.padding,
                         bias=orig.bias)

        new0.weight.data = orig.weight.data.sum(dim=1, 
                                                keepdim=True)
        vgg.features[0] = new0

        # freeze early layers, fine‐tune last conv block

        for name, param in vgg.features.named_parameters():
            layer_idx = int(name.split('.')[0])
            param.requires_grad = (layer_idx >= 24)

        self.features = vgg.features
        self.attn     = SpatialAttention(kernel_size=7)
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)           
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.pool(x)
        return self.classifier(x)

# -=-=-=-=-=-=- Instantiate -=-=-=-=-=-=-

model     = FERModel(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(

                    filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=LR, 
                    weight_decay=WEIGHT_DECAY
                    
                    )
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

# -=-=-=-=-=-=- CSV Logging -=-=-=-=-=-=-

import csv
from tqdm import tqdm

CSV_PATH = os.path.join(sys.path[0], 'trainingV2_log.csv')

with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

best_val_acc = 0.0

# -=-=-=-=-=-=- Training Loop -=-=-=-=-=-=-

for epoch in range(1, NUM_EPOCHS+1):
    
    model.train()
    t_loss = t_correct = t_total = 0

    train_bar = tqdm(train_loader,
                     desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]",
                     unit='batch')
    
    for imgs, labels in train_bar:

        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)

        loss = criterion(out, labels)
        loss.backward()

        optimizer.step()

        t_loss    += loss.item() * imgs.size(0)
        preds      = out.argmax(1)
        t_correct += (preds == labels).sum().item()
        t_total   += labels.size(0)

        train_bar.set_postfix({
            'loss': f"{t_loss/t_total:.4f}",
            'acc':  f"{100.*t_correct/t_total:.2f}%"
        })

    train_loss = t_loss / t_total
    train_acc  = 100. * t_correct / t_total

    # — Validation —

    model.eval()

    v_loss = v_correct = v_total = 0
    val_bar = tqdm(val_loader,
                   desc=f"Epoch {epoch}/{NUM_EPOCHS} [  Val ]",
                   unit='batch')
    
    with torch.no_grad():

        for imgs, labels in val_bar:

            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)

            v_loss    += loss.item() * imgs.size(0)
            preds      = out.argmax(1)
            v_correct += (preds == labels).sum().item()
            v_total   += labels.size(0)

            val_bar.set_postfix({
                'loss': f"{v_loss/v_total:.4f}",
                'acc':  f"{100.*v_correct/v_total:.2f}%"
            })

    val_loss = v_loss / v_total
    val_acc  = 100. * v_correct / v_total

    # — Log to CSV —
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.4f}", f"{train_acc:.2f}",
            f"{val_loss:.4f}",   f"{val_acc:.2f}"
        ])

    # — Check for best model & save/download if improved —

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        tqdm.write(f"🏆 New best model @ epoch {epoch} — Val Acc: {val_acc:.2f}%")
        if FileLink:
            display(FileLink(BEST_MODEL_PATH))

    # — Step scheduler —
    
    scheduler.step(val_acc)

print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")