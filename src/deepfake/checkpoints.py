from pathlib import Path
import torch
from .model import Detector


def save_atomic(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def load_detector(path, device='cpu'):
    if not Path(path).is_file():
        raise FileNotFoundError('No trained model found. Train first, then select best.pt or calibrated.pt.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('training_steps', 0) < 1 or checkpoint.get('class_names') != ['REAL', 'FAKE']:
        raise ValueError('Checkpoint is not a trained REAL/FAKE detector')
    model = Detector(**checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model, checkpoint
