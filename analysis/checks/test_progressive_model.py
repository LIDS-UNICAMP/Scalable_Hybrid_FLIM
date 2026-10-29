"""Model mechanics tests; local convs explicitly do NOT establish SPiFiL fitting."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from methods.autoencoder.progressive import EvolutionaryCNN, ProgressiveTrainer


class ProgressiveModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.config = {"image_size": 200, "device": "cpu", "lr": .001, "refine_lr": .0001}
        self.spec = {"channels": 4, "kernel": 3, "pool_stride": 2}
        self.x = torch.rand(2, 3, 200, 200)

    def block(self, channels_in, stride=2):
        return nn.Sequential(nn.Conv2d(channels_in, 4, 3, padding=1), nn.BatchNorm2d(4),
                             nn.ReLU(), nn.AvgPool2d(stride))

    def test_shapes_auxiliary_and_prefix(self):
        model = EvolutionaryCNN(self.config)
        block = self.block(3)
        initial = block[0].weight.detach().clone()
        model.add_block(block, self.spec)
        self.assertTrue(torch.equal(initial, block[0].weight))
        self.assertEqual(model.project(self.x).shape, (2, 1280, 100, 100))
        self.assertEqual(model.embed(self.x).shape, (2, 1280))
        self.assertEqual(model(self.x).shape, self.x.shape)
        self.assertTrue(torch.equal(model.decoders[0][1].blocks[0].bn1.weight, torch.ones(4)))
        with self.assertRaises(ValueError):
            model.add_block(self.block(4), {**self.spec, "connection": "concat"})

    def test_freeze_unfreeze_optimizer_and_checkpoint(self):
        model = EvolutionaryCNN(self.config).add_block(self.block(3), self.spec)
        trainer = ProgressiveTrainer(model, self.config)
        self.assertTrue(torch.isfinite(torch.tensor(trainer.train_epoch([self.x])["reconstruction_loss"])))
        old_weight = model.encoder[0][0].weight
        old_optimizer_state = trainer.optimizer.state[old_weight]["step"].clone()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "depth1.pt"
            trainer.save_checkpoint(first, depth=1)
            preserved = first.read_bytes()
            old_state = {k: v.clone() for k, v in model.encoder[0].state_dict().items()}
            model.add_block(self.block(4), self.spec)
            trainer.set_phase("A")
            self.assertEqual(trainer.optimizer.state[old_weight]["step"], old_optimizer_state)
            trainer.train_epoch([self.x])
            for key, value in old_state.items():
                self.assertTrue(torch.equal(value, model.encoder[0].state_dict()[key]), key)
            self.assertIsNone(old_weight.grad)
            self.assertIsNotNone(model.encoder[1][0].weight.grad)
            self.assertFalse(model.encoder[0][1].training)
            trainer.set_phase("B")
            trainer.train_epoch([self.x])
            self.assertIsNotNone(old_weight.grad)
            self.assertFalse(torch.equal(old_state["0.weight"], old_weight))
            self.assertTrue(model.encoder[0][1].training)
            self.assertTrue(first.read_bytes() == preserved)
            model.eval()
            with torch.no_grad():
                expected = model.embed(self.x)
                inference = model.inference_model()(self.x)
            self.assertTrue(torch.equal(expected, inference))
            second = Path(directory) / "depth2.pt"
            trainer.save_checkpoint(second, depth=2, checkpoint_id="synthetic-model-test")
            restored = ProgressiveTrainer(EvolutionaryCNN(self.config), self.config)
            metadata = restored.load_checkpoint(second)
            self.assertEqual(metadata["depth"], 2)
            self.assertEqual(restored.model.depth, 2)
            self.assertEqual(restored.epoch, trainer.epoch)
            restored.model.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(expected, restored.model.embed(self.x)))
            self.assertEqual(len(restored.optimizer.state), len(trainer.optimizer.state))
            restored.train_epoch([self.x])

    def test_ten_depths(self):
        model = EvolutionaryCNN(self.config)
        for i in range(10):
            stride = 2 if i < 3 else 1
            model.add_block(self.block(3 if i == 0 else 4, stride), {**self.spec, "pool_stride": stride})
        model.eval()
        with torch.no_grad():
            for depth in range(1, 11):
                self.assertEqual(model.projections[depth - 1].out_channels, 1280)
                self.assertEqual(model.embed(self.x[:1], depth).shape, (1, 1280))
            self.assertEqual(model(self.x[:1]).shape, self.x[:1].shape)
        with self.assertRaises(ValueError):
            model.add_block(self.block(4), self.spec)


if __name__ == "__main__":
    unittest.main()
