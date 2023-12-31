from torch.utils.data import Dataset

from lightning_accelerate.utils.config_utils import ConfigMixin


class DataModule(ConfigMixin):
    def prepare_data(self):
        """
        Contain downloading, saving and preparing data logics.
        This method is called only within a single process.
        """
    
    def setup(self):
        """
        Perform operations such as train/val/test split and transform data.
        This method is called on every GPU.
        """

    def get_training_dataset(self) -> Dataset:
        raise NotImplementedError
    
    def get_validation_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_test_dataset(self) -> Dataset:
        raise NotImplementedError
