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