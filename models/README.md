# Models Directory

This directory stores trained model checkpoints and the YuNet face detector.

## Directory Structure

```
models/
├── checkpoints/             # Trained PyTorch model files (.pt)
│   ├── mixed_frequency/     # CNN + Frequency model (controlled experiment)
│   │   ├── best.pt          # Best validation checkpoint
│   │   ├── calibrated.pt    # Temperature-calibrated inference checkpoint
│   │   └── last.pt          # Final epoch checkpoint
│   ├── mixed_cnn/           # CNN-only baseline model
│   │   ├── best.pt
│   │   ├── calibrated.pt
│   │   └── last.pt
│   └── final_frequency/     # Selected deployment model
│       └── calibrated.pt    # ← Used by the web app
│
└── assets/
    └── face_detection_yunet_2023mar.onnx  # YuNet ONNX face detector
```

## Checkpoint Format

Each `.pt` file is a dictionary saved with `torch.save()` containing:

| Key | Description |
|---|---|
| `model` | `state_dict` of the trained `Detector` |
| `config` | Architecture hyperparameters (backbone, mode, cutoff, patch_size) |
| `class_names` | `['REAL', 'FAKE']` |
| `training_steps` | Number of optimizer steps completed |
| `epoch` | Last completed epoch index |
| `history` | Per-epoch train loss and validation metrics |
| `temperature` | Calibration temperature (1.0 if uncalibrated) |
| `calibrated` | `True` if temperature scaling was applied |
| `manifest_sha256` | SHA-256 of the training manifest |

## Loading a Checkpoint

```python
from deepfake.checkpoints import load_detector

model, metadata = load_detector("models/checkpoints/final_frequency/calibrated.pt")
print(metadata['config'])      # Architecture config
print(metadata['temperature']) # Calibration temperature
```

## YuNet Face Detector

The ONNX face detector is automatically downloaded on first use by `deepfake/faces.py`. 
If offline, place the file manually at `models/assets/face_detection_yunet_2023mar.onnx`.

Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
