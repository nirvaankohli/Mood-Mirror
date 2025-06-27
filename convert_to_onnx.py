# convert_to_onnx.py

import torch
import torch.nn as nn
from torchvision import models

BEST_MODEL_PATH = "model(V2).pth"
ONNX_MODEL_PATH = "model(V2).onnx"
NUM_CLASSES     = 7
INPUT_SHAPE     = (1, 3, 64, 64)   # batch, channels, height, width
OPSET_VERSION   = 12               # >=11 is usually fine

model = models.efficientnet_b0(pretrained=False)
in_f  = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_f, NUM_CLASSES)

state_dict = torch.load(BEST_MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(INPUT_SHAPE, dtype=torch.float32)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_MODEL_PATH,
    export_params=True,            # store the trained parameter weights inside the model file
    opset_version=OPSET_VERSION,   
    do_constant_folding=True,      
    input_names=['input'],         
    output_names=['output'],       
    dynamic_axes={
        'input':  {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size'}
    }
)

print(f"✅ Exported ONNX model to: {ONNX_MODEL_PATH}")
