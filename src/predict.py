import argparse
import json
from pathlib import Path
from deepfake.predict import Predictor

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Predict one facial image and save its Grad-CAM overlay.')
    p.add_argument('image', help='Path to input image containing a single face')
    p.add_argument('--checkpoint', default='models/checkpoints/final_frequency/calibrated.pt',
                   help='Path to trained model checkpoint')
    p.add_argument('--output', default='results/predictions',
                   help='Directory to save original image, Grad-CAM heatmap, and prediction JSON')
    args = p.parse_args()
    result = Predictor(args.checkpoint).predict(args.image)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result.pop('original').save(out/'original.png')
    result.pop('overlay').save(out/'heatmap.png')
    (out/'prediction.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
