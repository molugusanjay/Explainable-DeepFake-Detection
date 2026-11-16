"""Prepare licensed FF++ videos using official source-disjoint split metadata."""
import argparse
import csv
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from deepfake.faces import FaceCropper
from deepfake.data import load_manifest

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def read_splits(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mapping, groups = {}, {}
    for split in ['train', 'val', 'test']:
        path = directory/f'{split}.json'
        if not path.exists():
            urllib.request.urlretrieve(
                f'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{split}.json', path)
        for pair in json.loads(path.read_text()):
            pair = [str(x).zfill(3) for x in pair]
            group = '_'.join(sorted(pair))
            for source in pair:
                if source in mapping and mapping[source] != split:
                    raise ValueError('Official split metadata contains overlapping sources')
                mapping[source], groups[source] = split, group
    return mapping, groups


def prepare(args):
    cv2.setNumThreads(2)
    root, out = Path(args.videos), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    mapping, groups = read_splits(out/'splits')
    cropper = FaceCropper()
    candidates = {}
    for method in ['real'] + METHODS:
        directory = (root/'original_sequences'/'youtube'/'c23'/'videos' if method == 'real'
                     else root/'manipulated_sequences'/method/'c23'/'videos')
        for path in sorted(directory.glob('*.mp4')):
            ids = path.stem.split('_')
            if any(x not in mapping for x in ids):
                continue
            splits = {mapping[x] for x in ids}
            if len(splits) != 1:
                raise ValueError(f'Cross-split source/target pair: {path}')
            if len({groups[x] for x in ids}) != 1:
                raise ValueError(f'Source pair does not match official grouping: {path}')
            split = splits.pop()
            candidates.setdefault((split, method), []).append((path, ids))
    if not candidates:
        raise FileNotFoundError('No FF++ C23 videos found. Check the dataset root folder.')
    rows, audit = [], Counter()
    for split in ['train', 'val', 'test']:
        for method in ['real'] + METHODS:
            items = candidates.get((split, method), [])[:]
            if not items:
                raise ValueError(f'No videos for {split}/{method}. Download more videos before proceeding.')
            random.Random(f'{args.seed}-{split}-{method}').shuffle(items)
            limit = args.train_videos if split == 'train' else args.eval_videos
            if limit > 0:
                items = items[:limit]
            nframes = args.train_frames if split == 'train' else args.eval_frames
            for path, ids in tqdm(items, desc=f'{split}/{method}'):
                audit[f'{split}/{method}/videos_selected'] += 1
                capture = cv2.VideoCapture(str(path))
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if count <= 0:
                    audit[f'{split}/{method}/unreadable_videos'] += 1
                for frame in np.unique(np.linspace(0, max(0, count-1), min(nframes, count), dtype=int)):
                    audit[f'{split}/{method}/frames_attempted'] += 1
                    relative = Path('faces')/split/method/path.stem/f'{frame:06d}.png'
                    target = out/relative
                    if not target.exists():
                        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                        ok, image = capture.read()
                        if not ok:
                            audit[f'{split}/{method}/unreadable_frames'] += 1
                            continue
                        try:
                            face, _ = cropper.crop(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                        except ValueError:
                            audit[f'{split}/{method}/face_rejected'] += 1
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        face.resize((224, 224), Image.Resampling.BILINEAR).save(target)
                    rows.append(dict(path=relative.as_posix(), label=int(method != 'real'), split=split,
                                     group=groups[ids[0]], source_ids='|'.join(ids), method=method,
                                     sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
                    audit[f'{split}/{method}/frames_saved'] += 1
                capture.release()
    manifest = out/'manifest.csv'
    if not rows:
        raise ValueError('No usable faces were extracted')
    for split in ['train', 'val', 'test']:
        present = {r['method'] for r in rows if r['split'] == split}
        if present != set(['real'] + METHODS):
            raise ValueError(f'{split} has a category with no usable faces; inspect source videos and increase the subset')
    with manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    load_manifest(manifest)
    report = dict(settings=vars(args), counts=dict(audit), frames=len(rows),
                  note='Video-source disjoint; not a verified identity-disjoint split. Inspect rejection rates.')
    (out/'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Prepared and checked {len(rows)} face crops: {manifest}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos', required=True)
    p.add_argument('--output', default='data')
    p.add_argument('--train-videos', type=int, default=120, help='Per class/method; 0 means all')
    p.add_argument('--eval-videos', type=int, default=30, help='Per class/method per split; 0 means all')
    p.add_argument('--train-frames', type=int, default=12)
    p.add_argument('--eval-frames', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.train_frames, args.eval_frames) < 1 or min(args.train_videos, args.eval_videos) < 0:
        p.error('Frame counts must be positive and video limits nonnegative')
    prepare(args)
