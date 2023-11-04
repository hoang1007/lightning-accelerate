import pytest
import tempfile

from lightning_accelerate import Trainer, TrainingArguments
from lightning_accelerate.dummy import DummyDataModule, DummyTrainingModule


def test_trainer_train():
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            tmp_dir, train_batch_size=2, eval_batch_size=1, num_epochs=1
        )

        tm = DummyTrainingModule()
        dm = DummyDataModule()

        Trainer(
            "temp", tm, args, dm
        ).fit()

def test_trainer_load_ckpt():
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            tmp_dir,
            train_batch_size=2,
            eval_batch_size=1,
            num_epochs=1,
            save_steps=1,
            save_total_limit=1,
            resume_from_checkpoint='latest'
        )
        tm = DummyTrainingModule()
        dtm = DummyDataModule()
        trainer = Trainer(
            "temp", tm, args, dtm
        )
        trainer.fit()
        trainer.load_state()
