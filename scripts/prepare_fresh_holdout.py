"""Freeze unused DFF source pairs before evaluating the final models."""
import csv
import hashlib
import io
import json
import random
import sys
import zipfile
from pathlib import Path
from collections import Counter
import cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from prepare_dff import read_pairs, member_map, METHODS
from deepfake.faces import FaceCropper


def main():
    root = Path(__file__).resolve().parent.parent
    out = root/'data/fresh_dff'
    out.mkdir(exist_ok=True)
    if (out/'manifest.csv').exists():
        raise ValueError('Frozen holdout exists; do not regenerate after seeing results')
    previous = list(csv.DictReader((root/'data/dff_prepared/manifest.csv').open()))
    excluded = {s for r in previous for s in r['source_ids'].split('|')}
    known = {r['sha256'] for folder in ['dff_prepared', 'prepared']
             for r in csv.DictReader((root/'data'/folder/'manifest.csv').open())}
    pairs = read_pairs(root/'data/dff_archives')
    archives = {m: zipfile.ZipFile(root/'data/dff_archives'/f'{m}.zip') for m in METHODS}
    members = {m: member_map(z) for m, z in archives.items()}
    common = set.intersection(*(set(m) for m in members.values()))
    candidates = sorted({pairs[k] for k in common if not set(pairs[k]) & excluded})
    random.Random(20261007).shuffle(candidates)
    selected = candidates[:300]
    (out/'selection.json').write_text(json.dumps(dict(seed=20261007, pairs=selected,
        policy='First 300 shuffled unused publisher pairs, fixed before final evaluation',
        excludes='Every source ID in all previous DFF train, validation and test splits'), indent=2))
    cv2.setNumThreads(2)
    cropper = FaceCropper()
    rows, rejected = [], Counter()
    for index, pair in enumerate(selected):
        pending = []
        for key in pair:
            if key not in common:
                continue
            for method, label in METHODS.items():
                with Image.open(io.BytesIO(archives[method].read(members[method][key]))) as im:
                    try:
                        face, box = cropper.crop(im.convert('RGB'))
                    except ValueError as error:
                        rejected[str(error)] += 1
                        continue
                target = out/'faces'/method/(key.replace('/', '__')+'.png')
                target.parent.mkdir(parents=True, exist_ok=True)
                face.resize((224, 224), Image.Resampling.BILINEAR).save(target)
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                pending.append(dict(path=target.relative_to(out).as_posix(), label=label,
                    split='test', group='fresh:'+hashlib.sha256('|'.join(pair).encode()).hexdigest()[:20],
                    source_ids='|'.join(pair), sha256=digest, method=method, dataset='DFF',
                    archive_member=members[method][key].filename))
        if any(r['sha256'] in known for r in pending):
            rejected['Entire source pair excluded because a crop duplicates previous data'] += len(pending)
        else:
            rows.extend(pending)
        if (index+1) % 50 == 0:
            print(f'Fresh holdout: {index+1}/300 pairs; {len(rows)} accepted', flush=True)
    with (out/'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(accepted=len(rows), rejected=dict(rejected), counts=dict(Counter(r['method'] for r in rows)),
        limitations=['Source-pair disjoint, not identity disjoint',
            'Exact crop duplicates checked; near-duplicate or uncropped-image duplication not exhaustively audited',
            'Only one dataset; this is not a new FaceForensics++ holdout'])
    (out/'preparation.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
