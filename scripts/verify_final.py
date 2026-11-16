"""Verify genuine Grad-CAM on original images, including selected error cases."""
import csv
import io
import json
import os
from pathlib import Path
import sys
import zipfile
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from deepfake.predict import Predictor


def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    run = root / 'models' / 'checkpoints' / 'final_frequency'
    out = run / 'demo'
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    originals = {}
    for domain, folder in [('DFF', 'dff_prepared'), ('FFPP', 'prepared')]:
        manifest_file = root / 'data' / folder / 'manifest.csv'
        if manifest_file.exists():
            for r in csv.DictReader(manifest_file.open()):
                originals[folder + '/' + r['path']] = dict(r, dataset=domain)
    predictions = list(csv.DictReader((run/'evaluation/test_predictions.csv').open()))
    ordered = sorted(predictions, key=lambda r: r['path'])
    chosen = [next(r for r in ordered if originals[r['path']]['dataset'] == domain and r['method'] == method)
              for domain, methods in [('DFF', ['wiki', 'inpainting', 'insight', 'text2img']),
                   ('FFPP', ['real', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'])]
              for method in methods]
    for label in ['0', '1']:
        error = next((r for r in ordered if r['label'] == label and
                      (float(r['probability_fake']) >= .5) != (label == '1')), None)
        if error and error not in chosen:
            chosen.append(error)
    predictor = Predictor(run/'calibrated.pt')
    records = []
    for number, selected in enumerate(chosen, 1):
        row = originals[selected['path']]
        if row['dataset'] == 'DFF':
            with zipfile.ZipFile(root/'data/dff_archives'/f"{row['method']}.zip") as z:
                with Image.open(io.BytesIO(z.read(row['archive_member']))) as im:
                    original = im.convert('RGB')
        else:
            parts = Path(row['path']).parts
            group, frame = parts[-2], int(Path(parts[-1]).stem)
            base = ('original_sequences/youtube' if row['method'] == 'real'
                    else 'manipulated_sequences/'+row['method'])
            cap = cv2.VideoCapture(str(root/'data/ffpp_videos'/base/'c23/videos'/(group+'.mp4')))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            cap.release()
            if not ok:
                raise ValueError('Unable to load original test frame')
            original = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = predictor.predict(original)
        image, overlay = result.pop('original'), result.pop('overlay')
        a, b = np.array(image), np.array(overlay)
        x0, y0, x1, y1 = result['face_box']
        outside = np.ones(a.shape[:2], bool)
        outside[y0:y1, x0:x1] = False
        assert np.array_equal(a[outside], b[outside])
        assert abs(result['probability_fake']+result['probability_real']-1) < 1e-5
        assert result['heatmap_class'] == result['prediction']
        image.save(out/f'{number}_original.png')
        overlay.save(out/f'{number}_heatmap.png')
        result.update(number=number, dataset=row['dataset'], method=row['method'],
            truth=['REAL', 'FAKE'][int(row['label'])], source_path=selected['path'],
            changed_pixels=int(np.any(a != b, axis=2).sum()),
            selection='First sorted test example per dataset and method, then first error per class')
        records.append(result)
        print(number, row['dataset'], row['method'], result['prediction'], flush=True)
    try:
        predictor.predict(Image.new('RGB', (256, 256), 'white'))
        raise AssertionError('Blank image accepted')
    except ValueError as error:
        assert 'No clear face' in str(error)
    (out/'verification.json').write_text(json.dumps(dict(examples=records,
        background_preserved=True, probabilities_sum_to_one=True,
        blank_rejected=True, class_specific_gradcam=True), indent=2))


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
