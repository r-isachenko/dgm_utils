"""Check AMP clipping with one scalar SGD step and a real CPU GradScaler."""
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch
from torch import nn

import dgm_utils.training as training


class _ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def loss(self, x):
        return {"total_loss": (self.weight * x).square().mean()}


class AMPGradientClippingTests(unittest.TestCase):
    def one_step(self, *, use_amp, clip, initial_scale=65536.0):
        model = _ScalarModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=initial_scale)
        self.assertTrue(scaler.is_enabled())
        # The helper uses CUDA autocast. These CPU tests check scaling/clipping,
        # so bypass only autocast while leaving GradScaler and SGD unchanged.
        with patch.object(training.torch.amp, "autocast", return_value=nullcontext()), \
             patch.object(training, "tqdm", side_effect=lambda batches, **kwargs: batches):
            stats = training.train_epoch(
                1, model, [torch.ones(1)], optimizer,
                gradient_clip_val=clip, device="cpu", use_amp=use_amp, scaler=scaler,
            )
        self.assertEqual(stats["total_loss"], [1.0])
        return model.weight.detach().item()

    def test_amp_clipping_is_independent_of_scale(self):
        # d(w^2)/dw=2; clip to 1, then SGD(lr=.1) must move w from 1 to .9.
        # Clipping the scaled gradient instead produces an update near .1/scale.
        for scale in (128.0, 65536.0):
            with self.subTest(initial_scale=scale):
                weight = self.one_step(use_amp=True, clip=1.0, initial_scale=scale)
                self.assertAlmostEqual(weight, 0.9, places=6)

    def test_amp_without_clipping(self):
        weight = self.one_step(use_amp=True, clip=None)
        self.assertAlmostEqual(weight, 0.8, places=6)

    def test_fp32_with_and_without_clipping(self):
        for clip, expected in ((1.0, 0.9), (None, 0.8)):
            with self.subTest(clip=clip):
                weight = self.one_step(use_amp=False, clip=clip)
                self.assertAlmostEqual(weight, expected, places=6)


if __name__ == "__main__":
    unittest.main()
