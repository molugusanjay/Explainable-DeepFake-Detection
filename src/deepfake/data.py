import csv
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN, STD = [.485, .456, .406], [.229, .224, .225]


def transform(train=False):
    ops = [T.Resize((224, 224))]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomApply([T.GaussianBlur(3, (.1, .7))], p=.15)]
    ops += [T.ToTensor()]
    if train:
        ops += [AddNoise()]
    ops += [T.Normalize(MEAN, STD)]
    return T.Compose(ops)


class AddNoise:
    def __call__(self, image):
        if torch.rand(()) < .15:
            image = (image + torch.randn_like(image)*.01).clamp(0, 1)
        return image


def load_manifest(path, required_splits=('train', 'val', 'test')):
    path = Path(path).resolve()
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    required = {'path', 'label', 'split', 'group', 'source_ids', 'sha256', 'method'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Manifest requires columns: {sorted(required)}')
    seen_groups, seen_sources, seen_hashes, seen_paths = {}, {}, {}, set()
    resolved_parents = {}
    for row in rows:
        split = row['split']
        if split not in ('train', 'val', 'test') or row['label'] not in ('0', '1'):
            raise ValueError('Invalid split or label')
        candidate = path.parent/row['path']
        # Resolving every frame's complete Windows path repeats expensive
        # junction checks. Resolve each parent once, while still following
        # file symlinks and enforcing containment for every image.
        parent = resolved_parents.setdefault(candidate.parent, None)
        if parent is None:
            parent = candidate.parent.resolve()
            resolved_parents[candidate.parent] = parent
        image_path = parent/candidate.name
        if image_path.is_symlink():
            image_path = image_path.resolve()
        if not image_path.is_relative_to(path.parent):
            raise ValueError('Image path must stay inside dataset directory')
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if image_path in seen_paths:
            raise ValueError('Duplicate manifest image path')
        seen_paths.add(image_path)
        if not row['group'] or not row['source_ids'] or not row['sha256']:
            raise ValueError('Missing provenance or content hash')
        for key, store in [(row['group'], seen_groups), (row['sha256'], seen_hashes)]:
            if key in store and store[key] != split:
                raise ValueError(f'Data leakage across splits: {key}')
            store[key] = split
        for source in row['source_ids'].split('|'):
            if source in seen_sources and seen_sources[source] != split:
                raise ValueError(f'Source video occurs in multiple splits: {source}')
            seen_sources[source] = split
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Image content changed: {image_path}')
        row['absolute_path'] = str(image_path)
    for split in required_splits:
        labels = {r['label'] for r in rows if r['split'] == split}
        if labels != {'0', '1'}:
            raise ValueError(f'{split} must contain both REAL and FAKE examples')
    return rows


class Faces(Dataset):
    def __init__(self, rows, split):
        self.rows = [r for r in rows if r['split'] == split]
        self.transform = transform(split == 'train')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['absolute_path']) as image:
            image = self.transform(image.convert('RGB'))
        return image, int(row['label'])
