"""Software correctness checks. These do not measure detector accuracy."""
import csv
import hashlib
import numpy as np
import pytest
import torch
from PIL import Image
from scipy.fft import dctn, idctn
from deepfake.model import frequency_residual, FGEM, Detector
from deepfake.explain import gradcam, face_overlay
from deepfake.data import load_manifest
from deepfake.metrics import metrics
from deepfake.checkpoints import load_detector

torch.set_num_threads(2)


def test_frequency_matches_independent_scipy_transform():
    rng = np.random.default_rng(1)
    array = rng.normal(size=(2, 3, 7, 7)).astype(np.float32)
    u, v = np.mgrid[:7, :7] / 6
    mask = (u + v)/2 > .25
    reference = idctn(dctn(array, axes=(-2, -1), norm='ortho')*mask, axes=(-2, -1), norm='ortho')
    actual = frequency_residual(torch.tensor(array)).numpy()
    np.testing.assert_allclose(actual, reference, atol=3e-6)


def test_low_frequency_constant_suppressed():
    assert frequency_residual(torch.ones(1, 2, 7, 7)).abs().max() < 2e-6


def test_patch_filter_has_no_cross_patch_leakage():
    module = FGEM(2, patch_size=7)
    x = torch.zeros(1, 2, 14, 14)
    x[:, :, 2, 3] = 1
    output = module.frequency(x)
    assert output[:, :, :7, :7].abs().sum() > 0
    assert torch.count_nonzero(output[:, :, 7:, :]) == 0
    assert torch.count_nonzero(output[:, :, :, 7:]) == 0


def test_detector_both_frequency_modules_receive_gradients():
    model = Detector(pretrained=False)
    logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 2)
    logits.sum().backward()
    for module in [model.patch, model.global_frequency]:
        for parameter in module.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert module.v.weight.grad.abs().sum() > 0


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cam_layer = torch.nn.Identity()

    def forward(self, x):
        value = self.cam_layer(x)
        return value[:, :2].mean((2, 3))


def test_gradcam_tracks_requested_class_and_cleans_hooks():
    model = ToyModel()
    x = torch.zeros(1, 3, 8, 8)
    x[:, 0, :4] = 1
    x[:, 1, 4:] = 1
    _, real, _ = gradcam(model, x, 0)
    _, fake, _ = gradcam(model, x, 1)
    assert real[:4].mean() > real[4:].mean()
    assert fake[4:].mean() > fake[:4].mean()
    assert len(model.cam_layer._forward_hooks) == 0


def test_overlay_keeps_background_and_zero_cam_unchanged():
    image = Image.new('RGB', (100, 100), (100, 100, 100))
    overlay, heat = face_overlay(image, (20, 20, 80, 80), np.ones((7, 7)), True)
    actual = np.array(overlay)
    assert np.array_equal(actual[:20], np.array(image)[:20])
    assert actual[50, 50, 0] > actual[50, 50, 1]
    assert heat[0, 0] == 1.0
    flat, _ = face_overlay(image, (20, 20, 80, 80), np.zeros((7, 7)), True)
    assert np.array_equal(flat, image)


def make_manifest(directory, leak=False):
    rows = []
    for split_index, split in enumerate(['train', 'val', 'test']):
        for label in [0, 1]:
            name = f'{split}_{label}.png'
            image = directory/name
            Image.new('RGB', (20, 20), (split_index*50, label*100, 10)).save(image)
            rows.append(dict(path=name, label=label, split=split, group=f'g{split_index}',
                             source_ids='same' if leak else f's{split_index}', method=str(label),
                             sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
    path = directory/'manifest.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_manifest_accepts_disjoint_sources(tmp_path):
    assert len(load_manifest(make_manifest(tmp_path))) == 6


def test_manifest_rejects_source_leakage(tmp_path):
    with pytest.raises(ValueError, match='multiple splits'):
        load_manifest(make_manifest(tmp_path, leak=True))


def test_manifest_rejects_modified_image(tmp_path):
    path = make_manifest(tmp_path)
    Image.new('RGB', (20, 20), 'white').save(tmp_path/'train_0.png')
    with pytest.raises(ValueError, match='content changed'):
        load_manifest(path)


def test_manifest_rejects_parent_directory_escape(tmp_path):
    folder = tmp_path/'data'
    folder.mkdir()
    path = make_manifest(folder)
    outside = tmp_path/'outside.png'
    outside.write_bytes((folder/'train_0.png').read_bytes())
    rows = list(csv.DictReader(path.open()))
    rows[0]['path'] = '../outside.png'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='inside dataset'):
        load_manifest(path)


def test_metrics_class_order_and_calibration():
    score = metrics([0, 0, 1, 1], [.1, .9, .8, .9])
    assert score['accuracy'] == .75
    assert score['confusion_matrix'] == [[1, 1], [0, 2]]
    assert score['recall_fake'] == 1


def test_untrained_checkpoint_is_rejected(tmp_path):
    path = tmp_path/'untrained.pt'
    torch.save({'training_steps': 0, 'class_names': ['REAL', 'FAKE']}, path)
    with pytest.raises(ValueError, match='not a trained'):
        load_detector(path)
