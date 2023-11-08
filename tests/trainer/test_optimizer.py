import pytest
import tempfile

from torch import nn
from lightning_accelerate import Trainer, TrainingArguments, TrainingModule
from lightning_accelerate.dummy import DummyDataModule, DummyTrainingModule


class BaseTrainingModule(TrainingModule):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(2, 1)
        self.net2 = nn.Linear(2, 1)

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        x, y = batch
        if optimizer_idx == 0:
            y_hat = self.net1(x)
        else:
            y_hat = self.net2(x)

        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y.float())
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = (self.net1(x) + self.net2(x)) / 2
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y.float())
        return loss


class ReturnIterableDict(DummyTrainingModule):
    def get_optim_params(self):
        return [
            {
                "params": self.parameters(),
                "lr_scale": 0.6,
            }
        ]


class ReturnIterable(DummyTrainingModule):
    def get_optim_params(self):
        return self.parameters()


class ReturnSequenceIterable(BaseTrainingModule):
    def get_optim_params(self):
        return [
            self.net1.parameters(),
            self.net2.parameters(),
        ]


class ReturnSequenceDict(BaseTrainingModule):
    def get_optim_params(self):
        return [
            [{"params": self.net1.parameters()}],
            [{"params": self.net2.parameters()}],
        ]


def test_create_optimizers():
    for tm_cls in (
        ReturnIterableDict,
        ReturnIterable,
        ReturnSequenceDict,
        ReturnSequenceIterable,
    ):
        tm = tm_cls()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir, train_batch_size=2, eval_batch_size=1, num_epochs=1
            )

            dm = DummyDataModule()

            Trainer("temp", tm, args, dm).fit()


def test_clip_grad():
    for tm_cls in (
        ReturnIterableDict,
        ReturnIterable,
        ReturnSequenceDict,
        ReturnSequenceIterable,
    ):
        tm = tm_cls()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                train_batch_size=2,
                eval_batch_size=1,
                num_epochs=1,
                max_grad_norm=1.0,
            )

            dm = DummyDataModule()

            Trainer("temp", tm, args, dm).fit()
