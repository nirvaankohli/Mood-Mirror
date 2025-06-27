import onnxruntime as ort
import numpy as np
from torchvision import transforms
import torch

class EmotionModel:
    def __init__(
        self,
        model_path: str = "models/model(V2).onnx",
        stress_class_idx: int = 3
    ):
        
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.stress_idx = stress_class_idx

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    def predict(self, frame: np.ndarray) -> float:

        x = self.preprocess(frame)                    
        x = x.unsqueeze(0).numpy().astype(np.float32)  

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: x}
        )
        probs = outputs[0]  


        stress_score = float(probs[0, self.stress_idx])
        return stress_score
