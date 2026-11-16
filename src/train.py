"""Train CNN + frequency features; save resumable checkpoints after each epoch."""
import argparse
import hashlib
import json
import random
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from deepfake.model import Detector
from deepfake.data import Faces, load_manifest
from deepfake.metrics import metrics
from deepfake.checkpoints import save_atomic
from deepfake.sampling import sample_weights, validation_score

torch.set_num_threads(2)
# Store torch hub downloads in the project models/assets directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('TORCH_HOME', str(_PROJECT_ROOT / 'models' / 'assets' / 'torch'))


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    logits, labels = [], []
    for images, target in loader:
        logits.append(model(images.to(device)).cpu())
        labels.append(target)
    return torch.cat(logits), torch.cat(labels)


def initialize_weights(model, path, config):
    """Transfer learned weights only; optimizer, calibration and history stay fresh."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('class_names') != ['REAL', 'FAKE'] or checkpoint.get('training_steps', 0) < 1:
        raise ValueError('Initialization requires a trained REAL/FAKE checkpoint')
    if checkpoint.get('config') != config:
        raise ValueError('Initialization requires the same architecture')
    model.load_state_dict(checkpoint['model'])
    return dict(checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                manifest_sha256=checkpoint.get('manifest_sha256'),
                epoch=checkpoint.get('epoch'), policy='Weights only; new optimizer and calibration')


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation < 1:
        raise ValueError('Epochs, batch size and accumulation must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}. Training a learned detector; no accuracy is assumed.')
    rows = load_manifest(args.manifest)
    digest = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    datasets = {s: Faces(rows, s) for s in ['train', 'val']}
    weights = sample_weights(datasets['train'].rows)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loaders = {s: DataLoader(datasets[s], batch_size=args.batch_size,
                            sampler=sampler if s == 'train' else None,
                            num_workers=args.workers, pin_memory=device == 'cuda') for s in datasets}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    resume = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    initial = getattr(args, 'init_checkpoint', None)
    if resume and initial:
        raise ValueError('Choose resume or transfer initialization, not both')
    config = dict(backbone=args.backbone, mode=args.mode, patch_size=7, cutoff=.25)
    if resume:
        if resume['manifest_sha256'] != digest or resume['config'] != config:
            raise ValueError('Resume requires the same dataset manifest and architecture')
    elif (out/'last.pt').exists():
        raise ValueError('Output already has a run. Resume it or choose a new output directory.')
    model = Detector(**config, pretrained=not args.no_pretrained and not bool(resume) and not bool(initial)).to(device)
    transfer = resume.get('transfer_parent') if resume else None
    if initial:
        transfer = initialize_weights(model, initial, config)
        print('Loaded prior detector weights; starting a new dataset experiment.', flush=True)
    
    # Split parameters to give custom FGEM layers a higher learning rate
    backbone_params, custom_params = [], []
    for name, param in model.named_parameters():
        if 'patch' in name or 'global_frequency' in name:
            custom_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr},
        {'params': custom_params, 'lr': args.lr * 10}
    ], betas=(.9, .999), weight_decay=.02)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    best, start, steps, stale = -1., 0, 0, 0
    history = []
    if resume:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        best, start, steps = resume['best_score'], resume['epoch']+1, resume['training_steps']
        stale, history = resume['stale'], resume['history']
        torch.set_rng_state(resume['torch_rng'])
        if device == 'cuda' and resume.get('cuda_rng'):
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        print(f'Resuming at epoch {start+1}; completed epochs will not repeat.')
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start, args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total = 0., 0
        progress = tqdm(loaders['train'], desc=f'Epoch {epoch+1}/{args.epochs}', mininterval=15)
        for index, (images, target) in enumerate(progress):
            images, target = images.to(device), target.to(device)
            group_start = (index//args.accumulation)*args.accumulation
            group_length = min(args.accumulation, len(loaders['train'])-group_start)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                loss = criterion(model(images), target)
            scaler.scale(loss/group_length).backward()
            if (index+1) % args.accumulation == 0 or index+1 == len(loaders['train']):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            total_loss += loss.item()*len(images)
            total += len(images)
            progress.set_postfix(loss=f'{total_loss/total:.4f}', refresh=False)
        logits, labels = collect(model, loaders['val'], device)
        score = metrics(labels.numpy(), logits.softmax(1)[:, 1].numpy())
        selection, domains = validation_score(datasets['val'].rows, logits.softmax(1)[:, 1].numpy())
        score['per_dataset'] = domains
        score['selection_score'] = selection
        improved = selection > best
        if improved:
            best, stale = selection, 0
        else:
            stale += 1
        history.append(dict(epoch=epoch+1, train_loss=total_loss/total, validation=score,
                            elapsed_seconds=time.perf_counter()-epoch_started))
        checkpoint = dict(model=model.state_dict(), config=config, optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), epoch=epoch, training_steps=steps,
                          best_score=best, stale=stale, history=history,
                          manifest_sha256=digest, class_names=['REAL', 'FAKE'],
                          temperature=1., calibrated=False, training_args=vars(args),
                          transfer_parent=transfer,
                          torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else [])
        save_atomic(checkpoint, out/'last.pt')
        if improved:
            save_atomic({k: v for k, v in checkpoint.items() if k not in
                         ('optimizer', 'scaler', 'torch_rng', 'cuda_rng')}, out/'best.pt')
        (out/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f"Validation accuracy {score['accuracy']:.2%}; balanced accuracy {score['balanced_accuracy']:.2%}", flush=True)
        if stale >= args.patience:
            print('Early stopping: validation score stopped improving.')
            break
    print(f'Finished. Best checkpoint: {out / "best.pt"}. Test-set performance has not been measured here.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/prepared/manifest.csv')
    p.add_argument('--output', default='results/runs/frequency')
    p.add_argument('--backbone', choices=['tiny', 'base'], default='tiny')
    p.add_argument('--mode', choices=['cnn', 'patch', 'global', 'both'], default='both')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=4e-5)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument('--resume')
    initialization.add_argument('--init-checkpoint', help='Transfer weights to a new dataset with a fresh optimizer')
    p.add_argument('--no-pretrained', action='store_true', help='For software tests only; normal training uses ImageNet weights')
    train(p.parse_args())
