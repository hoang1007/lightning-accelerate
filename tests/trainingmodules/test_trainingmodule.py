import pytest
from lightning_accelerate.dummy import DummyTrainingModule
import torch
from torch import nn

@pytest.mark.parametrize(
    "batch",
    [
        (torch.rand(2, 2), torch.randint(0, 10, (2, 1))),
        (torch.rand(1, 2), torch.randint(0, 10, (1, 1))),
    ],
)
def test_forward(batch):
    tm = DummyTrainingModule()
    tm.training_step(batch, 0, 0)
    tm.validation_step(batch, 0)
