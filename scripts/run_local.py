"""Local Windows/VS Code workflow. No Colab or Google Drive is required."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'download', 'prepare', 'train', 'calibrate', 'evaluate', 'app'])
    p.add_argument('--dataset', choices=['dff', 'ffpp'], default='dff')
    p.add_argument('--downloader', help='Path to the official downloader received after dataset approval')
    p.add_argument('--server', choices=['EU', 'EU2', 'CA'], default='EU2',
                   help='FaceForensics++ server; the access email specifies EU2')
    p.add_argument('--videos', default=str(ROOT/'data'/'ffpp_videos'))
    p.add_argument('--download-limit', type=int, default=300, help='Per category; 0 downloads all')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=2)
    args = p.parse_args()
    # Interpret a supplied relative downloader path in the caller's directory.
    downloader = Path(args.downloader).resolve() if args.downloader else None
    videos = Path(args.videos).resolve()
    os.chdir(ROOT)
    run = ROOT/'results'/'runs'/('dff_frequency' if args.dataset == 'dff' else 'local_frequency')
    manifest = ROOT/'data'/('dff_prepared' if args.dataset == 'dff' else 'prepared')/'manifest.csv'
    if args.action == 'check':
        import torch
        import torchvision
        result = {'python': sys.version.split()[0], 'torch': torch.__version__,
                  'torchvision': torchvision.__version__, 'cuda_available': torch.cuda.is_available(),
                  'cuda_build': torch.version.cuda}
        if torch.cuda.is_available():
            result.update(gpu=torch.cuda.get_device_name(0),
                          gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory/2**30, 1))
        print(json.dumps(result, indent=2))
        return
    if args.action == 'download':
        if args.dataset == 'dff':
            subprocess.run([sys.executable, str(ROOT/'scripts'/'download_dff.py')], check=True)
            return
        if downloader is None or not downloader.is_file():
            p.error('Supply --downloader with the official script received after dataset approval')
        if args.download_limit < 0:
            p.error('--download-limit must be nonnegative')
        for category in ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
            command = [sys.executable, str(downloader), str(videos),
                       '-d', category, '-c', 'c23', '-t', 'videos', '--server', args.server]
            if args.download_limit:
                command += ['--num_videos', str(args.download_limit)]
            # Keep the official terms prompt interactive; the user accepts it.
            subprocess.run(command, check=True)
        return
    if args.action == 'prepare':
        command = ([str(ROOT/'scripts'/'prepare_dff.py'), '--output', str(manifest.parent)] if args.dataset == 'dff'
                   else [str(ROOT/'scripts'/'prepare_ffpp.py'), '--videos', str(videos), '--output', str(manifest.parent)])
    elif args.action == 'train':
        import torch
        if not torch.cuda.is_available():
            p.error('GPU-enabled PyTorch is required for this local training profile. See LOCAL_VSCODE.md.')
        if not manifest.exists():
            p.error('Download and prepare the selected dataset before training')
        command = [str(ROOT/'src'/'train.py'), '--manifest', str(manifest), '--output', str(run),
                   '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
                   '--accumulation', str(args.accumulation), '--workers', '0']
        if (run/'last.pt').is_file():
            command += ['--resume', str(run/'last.pt')]
    elif args.action == 'calibrate':
        command = [str(ROOT/'src'/'evaluate.py'), '--calibrate', '--manifest', str(manifest),
                   '--checkpoint', str(run/'best.pt'), '--output', str(run),
                   '--batch-size', str(args.batch_size), '--workers', '0']
    elif args.action == 'evaluate':
        command = [str(ROOT/'src'/'evaluate.py'), '--manifest', str(manifest), '--checkpoint', str(run/'calibrated.pt'),
                   '--output', str(run/'evaluation'), '--batch-size', str(args.batch_size), '--workers', '0']
    else:
        command = [str(ROOT/'src'/'app.py'), '--checkpoint', str(run/'calibrated.pt')]
    subprocess.run([sys.executable, *command], check=True)


if __name__ == '__main__':
    main()
