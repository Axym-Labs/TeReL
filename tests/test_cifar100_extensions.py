import torch
import torch.nn.functional as F

from terel.resubmission.data import TemporalTensorDataset, encoder_batches, encoder_order
from terel.resubmission.locally_connected import MaskedLinear, receptive_field_mask
from terel.resubmission.training import augment_cifar_batch


def test_mask_matches_convolution_support():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(3, 4, 5, stride=2, padding=2, bias=False)
    mask, shape = receptive_field_mask((3, 8, 8), 4, 5, 2, 2)
    basis = torch.eye(3 * 64).reshape(-1, 3, 8, 8)
    dense = conv(basis).reshape(3 * 64, -1).T
    assert shape == (4, 4, 4)
    assert torch.equal(dense.abs() > 0, mask)


def test_masked_linear_gradient_stays_in_receptive_field():
    mask, _ = receptive_field_mask((3, 8, 8), 2, 3, 2, 1)
    layer = MaskedLinear(mask)
    layer(torch.randn(5, 3 * 64)).square().sum().backward()
    assert torch.all(layer.weight.grad[~mask] == 0)


def test_repeated_views_order_and_boundaries():
    data = TemporalTensorDataset(
        features=torch.arange(5, dtype=torch.float32)[:, None],
        labels=torch.zeros(5, dtype=torch.long),
        boundaries=torch.zeros(5, dtype=torch.bool),
    )
    index, boundaries = encoder_order(data, order_mode="repeated_views", seed=0, chunk_size=3)
    assert len(index) == 15
    assert torch.equal(index.reshape(5, 3), index.reshape(5, 3)[:, :1].expand(5, 3))
    assert boundaries.tolist() == [True, False, False] * 5
    batches = list(encoder_batches(data, batch_size=1, order_mode="repeated_views", seed=0, chunk_size=3))
    assert len(batches) == 15


def test_cifar_crop_flip_is_a_shifted_or_flipped_crop():
    images = torch.randn(64, 3, 32, 32)
    augmented = augment_cifar_batch(images.reshape(64, -1), seed=3).reshape(64, 3, 32, 32)
    padded = F.pad(images, (4, 4, 4, 4))
    for image, padded_image in zip(augmented, padded):
        crops = [
            padded_image[:, y : y + 32, x : x + 32]
            for y in range(9)
            for x in range(9)
        ]
        assert any(torch.equal(image, c) or torch.equal(image, c.flip(2)) for c in crops)


def test_standardized_probe_ignores_constant_features():
    import torch
    from terel.resubmission.evaluation import fit_linear_probe

    torch.manual_seed(0)
    x = torch.randn(400, 4)
    x[:, 3] = 0.0  # constant in training
    y = (x[:, 0] > 0).long()
    probe, _ = fit_linear_probe(x, y, num_classes=2, seed=0, epochs=30, batch_size=50,
                                optimizer_name="plain_sgd", learning_rate=0.1, weight_decay=0.0,
                                device=torch.device("cpu"), standardize=True)
    test = torch.randn(200, 4)
    test[:, 3] = 1.0  # active only at evaluation
    logits = probe(test)
    assert torch.isfinite(logits).all() and logits.abs().max() < 100
    assert (logits.argmax(1) == (test[:, 0] > 0).long()).float().mean() > 0.9
