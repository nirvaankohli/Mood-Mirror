#!/usr/bin/env python3
import sys
import os


import torch
from timm import create_model

# ——— Config ———
BEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model(V4).pth')
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model(V4).onnx')

# must match your training
MODEL_NAME  = 'tf_efficientnetv2_s.in21k'
NUM_CLASSES = 7
IMG_SIZE    = 112  # the H and W used during training

def export_to_onnx():

    # build model
    model = create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    # load weights
    checkpoint = torch.load(BEST_MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # dummy input
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device='cpu')

    # export
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
    print(f"✅ ONNX model saved to: {ONNX_MODEL_PATH}")

if __name__ == '__main__':
    export_to_onnx()
