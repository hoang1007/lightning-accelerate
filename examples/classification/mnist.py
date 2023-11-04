from torch.utils.data import Dataset
from lightning_accelerate import TrainingArguments, Trainer, TrainingModule, DataModule
from lightning_accelerate.metrics import MeanMetric, Accuracy

from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms


class MnistTrainingModule(TrainingModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        self.accum_loss = MeanMetric()
        self.accuracy = Accuracy()

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        x, y = batch
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)

        self.accum_loss.update(loss.item())
        self.log({"train/loss_step": self.accum_loss.compute()})
        return loss

    def on_train_epoch_end(self):
        print("TRAIN END EPOCH")
        self.log({"train/loss_epoch": self.accum_loss.compute()})
        self.accum_loss.reset()

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.model(x)
        pred = logits.argmax(dim=-1)
        self.accuracy.update(pred, y)

    def on_validation_epoch_end(self):
        self.log({"val/acc": self.accuracy.compute()})
        self.accuracy.reset()

    def get_optim_params(self):
        return [self.model.parameters()]


class MnistDataModule(DataModule):
    def prepare_data(self):
        # Place downloading data to avoid downloading data in every process.
        train_data = MNIST("root", train=True, download=True)
        val_data = MNIST("root", train=False, download=True)

    def get_training_dataset(self) -> Dataset:
        return MNIST(
            "root",
            train=True,
            transform=transforms.Compose([
                transforms.RandomAffine(15), transforms.ToTensor()
            ]),
        )

    def get_validation_dataset(self) -> Dataset:
        return MNIST("root", train=False, transform=transforms.ToTensor())


def main():
    args = TrainingArguments(
        experiment_name="mnist-classification",
        overwrite_output_dir=True,
        train_batch_size=8,
        num_epochs=1,
        resume_from_checkpoint="latest",
        # tracker='wandb'
        tracker=None
    )
    data_module = MnistDataModule()
    training_module = MnistTrainingModule()

    Trainer(
        "lightning-accelerate",
        training_module=training_module,
        training_args=args,
        data_module=data_module,
    ).fit()

if __name__ == "__main__":
    main()
