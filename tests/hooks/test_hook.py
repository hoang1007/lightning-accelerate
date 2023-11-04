import sys

sys.path.append("..")

import pytest
import tempfile

from lightning_accelerate import Trainer, TrainingArguments
from lightning_accelerate.dummy import DummyDataModule, DummyTrainingModule


class TrainingModuleWithHook(DummyTrainingModule):
    def on_start(self):
        self.on_start_called = True
        return super().on_start()

    def on_end(self):
        self.on_end_called = True
        return super().on_end()

    def on_train_epoch_start(self):
        self.on_train_epoch_start_called = True
        return super().on_train_epoch_start()

    def on_train_epoch_end(self):
        self.on_train_epoch_end_called = True
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self):
        self.on_validation_epoch_start_called = True
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        self.on_validation_epoch_end_called = True
        return super().on_validation_epoch_end()

    def on_train_batch_start(self):
        self.on_train_batch_start_called = True
        return super().on_train_batch_start()

    def on_train_batch_end(self):
        self.on_train_batch_end_called = True
        return super().on_train_batch_end()

    def on_validation_batch_start(self):
        self.on_validation_batch_start_called = True
        return super().on_validation_batch_start()

    def on_validation_batch_end(self):
        self.on_validation_batch_end_called = True
        return super().on_validation_batch_end()


def test_hook():
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            tmp_dir, train_batch_size=2, eval_batch_size=1, num_epochs=1, eval_steps=0.9
        )

        tm = TrainingModuleWithHook()
        dm = DummyDataModule()

        Trainer("temp", tm, args, dm).fit()

        assert tm.on_start_called
        assert tm.on_end_called
        assert tm.on_train_epoch_start_called
        assert tm.on_train_epoch_end_called
        assert tm.on_validation_epoch_start_called
        assert tm.on_validation_epoch_end_called
        assert tm.on_train_batch_start_called
        assert tm.on_train_batch_end_called
        assert tm.on_validation_batch_start_called
        assert tm.on_validation_batch_end_called
