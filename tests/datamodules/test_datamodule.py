import pytest

from lightning_accelerate.datamodules import DataModule
import torch
from torch.utils.data import TensorDataset

class MyDataModule(DataModule):
    def prepare_data(self):
        pass

    def setup(self):
        self.train_data = TensorDataset(torch.rand(10, 2))
        self.val_data = TensorDataset(torch.rand(10, 2))
    
    def get_training_dataset(self):
        return self.train_data

    def get_validation_dataset(self):
        return self.val_data

def test_datamodule():
    with pytest.raises(NotImplementedError):
        DataModule().get_training_dataset()
    with pytest.raises(NotImplementedError):
        DataModule().get_validation_dataset()
    
    dm = MyDataModule()
    dm.prepare_data()
    dm.setup()

    assert len(dm.get_training_dataset()) == 10
    assert len(dm.get_validation_dataset()) == 10
    assert len(dm.get_training_dataset()[0][0]) == 2
    assert len(dm.get_validation_dataset()[0][0]) == 2
