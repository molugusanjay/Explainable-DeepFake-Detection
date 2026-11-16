"""Actual gradient-based class activation maps with face-restricted display."""
import cv2
import numpy as np
import torch
from PIL import Image


def gradcam(model, tensor, target_class=None):
    activations = []
    handle = model.cam_layer.register_forward_hook(lambda module, inputs, output: activations.append(output))
    try:
        model.eval()
        with torch.enable_grad():
            tensor = tensor.detach().requires_grad_(True)
            logits = model(tensor)
            target_class = int(logits[0].argmax()) if target_class is None else int(target_class)
            gradient = torch.autograd.grad(logits[0, target_class], activations[0])[0]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights*activations[0]).sum(1))[0]
            cam = cam.detach().float().cpu().numpy()
        maximum = float(cam.max())
        if maximum > 1e-10:
            cam /= maximum
        else:
            cam = np.zeros_like(cam)
        return logits.detach(), cam, target_class
    finally:
        handle.remove()


def face_overlay(image, box, cam, fake=True):
    rgb = np.asarray(image.convert('RGB')).copy()
    x0, y0, x1, y1 = box
    h, w = y1-y0, x1-x0
    heat = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    # Show the full Grad-CAM output rather than restricting to an artificial facial oval.
    # This allows it to work on side faces and full non-face random images.
    alpha = (.60 * heat * (heat >= .35))[..., None]
    color = np.zeros((h, w, 3), dtype=np.float32)
    color[:] = (255, 48, 35) if fake else (40, 130, 255)
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    rgb[y0:y1, x0:x1] = np.clip(crop*(1-alpha) + color*alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb), heat


def explain_regions(heat, predicted, uses_frequency=True):
    if float(heat.max()) <= 1e-10:
        return ('Grad-CAM did not produce a positive spatial activation for this prediction. '
                'No facial region can be highlighted meaningfully for this image.')
    h, w = heat.shape
    vertical = ['upper', 'middle', 'lower']
    horizontal = ['left', 'central', 'right']
    strengths = []
    for i, ys in enumerate(np.array_split(np.arange(h), 3)):
        for j, xs in enumerate(np.array_split(np.arange(w), 3)):
            strengths.append((float(heat[np.ix_(ys, xs)].mean()), f'{vertical[i]} {horizontal[j]} facial area'))
    strongest = [name for value, name in sorted(strengths, reverse=True)[:2] if value > 0]
    regions = ' and '.join(strongest)
    method = ('The classifier combines learned visual features with mid/high-frequency feature information. '
              if uses_frequency else 'This baseline classifier uses learned visual features only. ')
    return (f'The {regions} contributed most strongly in the displayed map to the {predicted} prediction '
            '(left/right refer to the image as displayed). '
            + method +
            'Grad-CAM shows model attention; it does not prove those pixels were edited or identify a specific artifact type.')
