import torch
import pytest
from train import initialize_weights


def test_transfer_copies_weights_without_training_state(tmp_path):
    source, target = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    config = {'backbone': 'tiny', 'mode': 'both', 'patch_size': 7, 'cutoff': .25}
    path = tmp_path/'prior.pt'
    torch.save(dict(model=source.state_dict(), config=config, class_names=['REAL', 'FAKE'],
                    training_steps=100, epoch=3, manifest_sha256='prior-dataset',
                    calibrated=True, temperature=2.2, optimizer={'old': True}), path)
    provenance = initialize_weights(target, path, config)
    assert torch.equal(source.weight, target.weight)
    assert torch.equal(source.bias, target.bias)
    assert provenance['manifest_sha256'] == 'prior-dataset'
    assert 'optimizer' not in provenance and 'temperature' not in provenance
    with pytest.raises(ValueError, match='same architecture'):
        initialize_weights(target, path, {**config, 'mode': 'cnn'})
