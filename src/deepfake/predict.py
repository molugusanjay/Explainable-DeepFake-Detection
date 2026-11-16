import threading
import torch
from .checkpoints import load_detector
from .data import transform
from .faces import FaceCropper, read_rgb
from .explain import gradcam, face_overlay, explain_regions


class Predictor:
    def __init__(self, checkpoint, device=None):
        torch.set_num_threads(2)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.metadata = load_detector(checkpoint, self.device)
        self.cropper = FaceCropper()
        self.transform = transform(False)
        self.lock = threading.Lock()

    def predict(self, image):
        if image is None:
            raise ValueError('Upload a facial image first.')
        image = read_rgb(image)
        with self.lock:
            face, box = self.cropper.crop(image)
            tensor = self.transform(face).unsqueeze(0).to(self.device)
            logits, cam, predicted = gradcam(self.model, tensor)
            temperature = self.metadata.get('temperature', 1.)
            probability = (logits/temperature).softmax(1)[0].cpu().numpy()
        label = ['REAL', 'FAKE'][predicted]
        overlay, heat = face_overlay(image, box, cam, fake=predicted == 1)
        confidence = float(probability[predicted])
        explanation = explain_regions(heat, label, uses_frequency=self.metadata['config']['mode'] != 'cnn')
        if confidence < .70:
            explanation = 'Low confidence: treat this prediction as uncertain. ' + explanation
        calibration = 'Validation-calibrated' if self.metadata.get('calibrated') else 'Uncalibrated'
        return dict(prediction=label, confidence=confidence, probability_fake=float(probability[1]),
                    probability_real=float(probability[0]), confidence_type=calibration,
                    explanation=explanation, original=image, overlay=overlay,
                    face_box=list(box), heatmap_class=label,
                    display_mask='Approximate facial oval, not a pixel-level forgery mask')
