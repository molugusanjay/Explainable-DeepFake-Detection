"""Download the public DeepFakeFace release and verify the publisher's SHA-256 hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REVISION = 'eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c'
FILES = {
    'wiki.zip': (1766572515, 'e83b55d6692d54e5aad38427531fa831bf30d1051ccc4267d4db90f03aaa7136'),
    'inpainting.zip': (1082027665, '795b751f0967b2d940fbd9825e05c541666ec6f8c634acc7640044542897b7b2'),
    'insight.zip': (1081375437, '3265a6f1eed49c55068c88929608d31c38db4588d942b9679b5f5ed1479ba5d7'),
    'text2img.zip': (1134392707, '95e628873d1daea2b539a6bdf5036728a6879470ac37d6a487e091208db28f3b'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(name, directory):
    expected_size, expected_hash = FILES[name]
    target = Path(directory)/name
    if target.exists():
        if target.stat().st_size == expected_size and sha256(target) == expected_hash:
            print(f'{name}: already downloaded and verified', flush=True)
            return
        raise ValueError(f'Existing archive failed verification: {target}. Preserve it and investigate.')
    temporary = target.with_suffix('.zip.part')
    url = f'https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/{REVISION}/{name}?download=true'
    for attempt in range(3):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset == expected_size:
                break
            headers = {'User-Agent': 'DeepFakeFace-academic-project/1.0'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Download server returned an unexpected byte range')
                if not append:
                    offset = 0
                last_report = offset
                with temporary.open('ab' if append else 'wb') as f:
                    while chunk := response.read(1024*1024):
                        f.write(chunk)
                        offset += len(chunk)
                        if offset-last_report >= 128*1024*1024:
                            print(f'{name}: {offset/expected_size:.0%}', flush=True)
                            last_report = offset
            if offset != expected_size:
                raise ValueError(f'Incomplete archive: {name}, {offset}/{expected_size} bytes')
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)
    if temporary.stat().st_size != expected_size or sha256(temporary) != expected_hash:
        raise ValueError(f'Archive checksum mismatch: {temporary}')
    temporary.replace(target)
    print(f'{name}: SHA-256 verified', flush=True)


def download(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda name: download_one(name, directory), FILES))
    (directory/'provenance.json').write_text(json.dumps(dict(
        dataset='OpenRL/DeepFakeFace', revision=REVISION, files=FILES,
        source='https://huggingface.co/datasets/OpenRL/DeepFakeFace',
        note='Use for this academic experiment; consult original dataset terms. Do not redistribute the dataset in project exports.'
    ), indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/dff_archives')
    download(p.parse_args().output)
