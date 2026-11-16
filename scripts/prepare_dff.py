"""Prepare a reproducible source-pair-disjoint DeepFakeFace image subset.

Grouping follows the publisher's actual source/target pairs. This is NOT an
identity-disjoint benchmark: different photographs of a celebrity can recur.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import pickle
import random
import urllib.request
import zipfile

import cv2
import numpy as np
from PIL import Image
# Allow direct execution without depending on an editable install of another checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from deepfake.data import load_manifest
from deepfake.faces import FaceCropper
from download_dff import FILES, REVISION, sha256

METADATA_URL = 'https://raw.githubusercontent.com/OpenRL-Lab/DeepFakeFace/main/data/wiki.pickle'
METADATA_SHA256 = '9d41f13fabf80ce356cc677efc694d9f831114e765c785b3de9ed5ea3e7640f1'
METHODS = {'wiki': 0, 'inpainting': 1, 'insight': 1, 'text2img': 1}


class MetadataOnly(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy', 'dtype'): np.dtype,
                   ('numpy.core.multiarray', 'scalar'): np._core.multiarray.scalar,
                   ('numpy._core.multiarray', 'scalar'): np._core.multiarray.scalar}
        if (module, name) not in allowed:
            raise ValueError(f'Unexpected object in dataset metadata: {module}.{name}')
        return allowed[(module, name)]


def read_pairs(directory):
    path = directory/'wiki_metadata.pickle'
    if not path.exists():
        temporary = path.with_suffix('.part')
        urllib.request.urlretrieve(METADATA_URL, temporary)
        if sha256(temporary) != METADATA_SHA256:
            raise ValueError('Publisher pairing metadata changed; review before use')
        temporary.replace(path)
    if sha256(path) != METADATA_SHA256:
        raise ValueError('Pairing metadata checksum mismatch')
    with path.open('rb') as f:
        paths, prompts, pairs = MetadataOnly(f).load()
    mapping = {}
    for first, second in pairs:
        pair = tuple(sorted((str(first), str(second))))
        for item in pair:
            if item in mapping and mapping[item] != pair:
                raise ValueError('Source occurs in inconsistent pair metadata')
            mapping[item] = pair
    if set(map(str, paths)) != set(mapping):
        raise ValueError('Incomplete source-pair metadata')
    return mapping


def member_map(archive):
    result = {}
    for info in archive.infolist():
        p = PurePosixPath(info.filename)
        if info.is_dir() or p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        if len(p.parts) < 2 or '..' in p.parts or info.file_size > 50*1024*1024:
            raise ValueError(f'Unexpected image member: {info.filename}')
        key = '/'.join(p.parts[-2:])
        if key in result:
            raise ValueError(f'Duplicate source filename: {key}')
        result[key] = info
    return result


def prepare(args):
    cv2.setNumThreads(2)
    root, out = Path(args.archives), Path(args.output)
    if (out/'manifest.csv').exists():
        load_manifest(out/'manifest.csv')
        print('Existing manifest verified; use another output folder to change the experiment.')
        return
    out.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(root)
    archives = {}
    try:
        for name, (size, digest) in FILES.items():
            path = root/name
            if path.stat().st_size != size or sha256(path) != digest:
                raise ValueError(f'Archive failed verification: {path}')
            archives[path.stem] = zipfile.ZipFile(path)
        members = {method: member_map(z) for method, z in archives.items()}
        common = set.intersection(*(set(m) for m in members.values()))
        if not common or not common.issubset(pairs):
            raise ValueError('Archive filenames do not match publisher pair metadata')
        groups = sorted(set(pairs[k] for k in common))
        random.Random(args.seed).shuffle(groups)
        groups = groups[:args.groups] if args.groups else groups
        if len(groups) < 30:
            raise ValueError('At least 30 source-pair groups are required')
        print(f'Archive images: { {k: len(v) for k,v in members.items()} }; shared filenames: {len(common)}', flush=True)
        print(f'Preparing {len(groups)} source-pair groups with every available category.', flush=True)
        cropper = FaceCropper()
        rows, rejected = [], Counter()
        # Union exact raw/pixel/crop duplicates BEFORE assigning train/val/test.
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            a, b = find(a), find(b)
            parent[max(a, b)] = min(a, b)
        hashes = {}
        for index, pair in enumerate(groups):
            group = hashlib.sha256('|'.join(pair).encode()).hexdigest()[:20]
            find(group)
            for key in pair:
                if key not in common:
                    continue
                for method, label in METHODS.items():
                    raw = archives[method].read(members[method][key])
                    with Image.open(io.BytesIO(raw)) as image:
                        rgb = image.convert('RGB')
                        pixel_hash = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                        try:
                            face, box = cropper.crop(rgb)
                        except ValueError as e:
                            rejected[f'{method}: {e}'] += 1
                            continue
                    relative = Path('faces')/method/(key.replace('/', '__')+'.png')
                    destination = out/relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    face.resize((224, 224), Image.Resampling.BILINEAR).save(destination)
                    digest = sha256(destination)
                    for value in (hashlib.sha256(raw).hexdigest(), pixel_hash, digest):
                        if value in hashes:
                            union(group, hashes[value])
                        hashes[value] = group
                    rows.append(dict(path=relative.as_posix(), label=label, group=group,
                                     source_ids='|'.join(pair), sha256=digest, method=method,
                                     archive_member=members[method][key].filename,
                                     original_box=','.join(map(str, box))))
            if (index+1) % 100 == 0 or index+1 == len(groups):
                print(f'Prepared {index+1}/{len(groups)} groups; accepted {len(rows)} images; rejected {sum(rejected.values())}', flush=True)
        components = sorted(set(find(r['group']) for r in rows))
        random.Random(args.seed).shuffle(components)
        n = len(components)
        splits = {g: ('train' if i < int(.8*n) else 'val' if i < int(.9*n) else 'test')
                  for i, g in enumerate(components)}
        for row in rows:
            row['group'] = find(row['group'])
            row['split'] = splits[row['group']]
        with (out/'manifest.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        load_manifest(out/'manifest.csv')
        report = dict(dataset='OpenRL/DeepFakeFace', revision=REVISION, seed=args.seed,
                      selected_source_pairs=len(groups), independent_components=n,
                      accepted=len(rows), rejected=dict(rejected),
                      counts=dict(Counter(f"{r['split']}/{r['method']}" for r in rows)),
                      pairing_metadata_sha256=METADATA_SHA256,
                      grouping='Publisher source/target pairs plus exact image duplicate components',
                      limitations=['Not identity-disjoint: celebrities may recur in different photographs.',
                                   'Near-duplicate photographs are not exhaustively removed.',
                                   'A reproducible subset, not the complete official dataset or a published benchmark split.',
                                   'Only images passing the single-face detector are evaluated.'])
        (out/'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for archive in archives.values():
            archive.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives', default='data/dff_archives')
    p.add_argument('--output', default='data/dff_prepared')
    p.add_argument('--groups', type=int, default=1500, help='0 uses all source pairs; local default is a subset')
    p.add_argument('--seed', type=int, default=42)
    prepare(p.parse_args())
