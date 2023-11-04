# Contains dummy functions for testing purposes

from typing import Iterable, List
import torch
from torch.utils.data import TensorDataset
from lightning_accelerate import DataModule, TrainingModule
from lightning_accelerate.hooks import BaseHook


class DummyDataModule(DataModule):
    def prepare_data(self):
        self.prepare_data_called = True
        return super().prepare_data()

    def setup(self):
        self.setup_called = True
        return super().setup()

    def get_training_dataset(self):
        return TensorDataset(torch.randn(10, 2), torch.randint(0, 2, (10, 1)))

    def get_validation_dataset(self):
        return TensorDataset(torch.randn(10, 2), torch.randint(0, 2, (10, 1)))

    def get_test_dataset(self):
        return TensorDataset(torch.randn(10, 2), torch.randint(0, 2, (10, 1)))


class DummyTrainingModule(TrainingModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        x, y = batch
        y_hat = self.net(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y.float())
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self.net(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y.float())
        return loss

    def get_optim_params(self):
        return [self.net.parameters()]
