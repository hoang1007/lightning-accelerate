**Lightning Accelerate** aim to provide a simple and easy-to-use framework for training deep learning model on GPU, TPU, etc... with 🤗 [Huggingface's Accelerate](https://github.com/huggingface/accelerate) and⚡️[Pytorch Lightning](https://github.com/Lightning-AI/lightning)'s style.

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
  - [Training](#training)
  - [Training with advanced features](#training-with-advanced-features)
  - [Evaluation](#evaluation)
- [Contributing](#contributing)
- [Acknowledgement](#acknowledgement)

## Installation
To install **Lightning Accelerate**, run this command:
```bash
pip install git+https://github.com/hoang1007/lightning-accelerate.git
```

## Features
- [x] Support training with multiple GPUs, TPUs, etc...
- [x] Support finetuning models efficiently with [LoRA](https://arxiv.org/abs/2106.09685)
- [x] Support several optimization techniques such as mixed precision, DeepSpeed, bitandbytes, etc...
- [x] Support tracking experiment with [Wandb](https://wandb.ai/site) and [Tensorboard](https://www.tensorflow.org/tensorboard) 

## Usage
### Training
To train a model, you need to define a `TrainingModule` and a `DataModule`. Here is an simple example of training a digit classifier on MNIST dataset:
```python
# -------------------
# Step 1: Define a TrainingModule.
# This module contains the model, training and evaluation logics to easy training with `Trainer` later.
# -------------------
class MnistTrainingModule(TrainingModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        x, y = batch
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        return loss

    def get_optim_params(self):
        return [self.model.parameters()]

# -------------------
# Step 2: Define a DataModule. This module contains the data preparation logics such as downloading data, preprocessing, etc... and then is used to feed to the `TrainingModule` for training and evaluation.
# -------------------
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

# -------------------
# Step 3: Configure parameters with `TrainingArguments` and start training!
# -------------------
args = TrainingArguments("mnist", train_batch_size=32, num_epochs=10)
training_module = MnistTrainingModule()
data_module = MnistDataModule()

Trainer(
    training_module=training_module,
    training_args=args,
    data_module=data_module,
).fit()
```

### Training with advanced features
You can accelerate the training process with several techniques such as mixed precision, DeepSpeed, etc... which are supported by `Accelerate`. For details, please refer to [Accelerate's documentation](https://huggingface.co/docs/accelerate/).
For example, to train your models on multiple GPUs, you can run
```bash
accelerate launch --multi_gpu my_script.py
```

### Evaluation
To evaluate the pretrained model, you can use `Trainer.evaluate` method:
```python
args = TrainingArguments(
    "mnist",
    eval_batch_size=32,
    # Set `resume_from_checkpoint` to the path of the checkpoint you want to evaluate or set to `latest` to evaluate the latest checkpoint.
    resume_from_checkpoint='latest'
)
training_module = MnistTrainingModule()
data_module = MnistDataModule()

# Trainer will automatically load the checkpoint and evaluate the model.
Trainer(
    training_module=training_module,
    training_args=args,
    data_module=data_module,
).evaluate()
```

I build the framework on top of [Huggingface's Accelerate](https://github.com/huggingface/accelerate) with minimum requirements while maintaining the code style as similar as possible to [Pytorch Lightning](https://github.com/Lightning-AI/lightning) 😊.

## Contributing
I am an inexperienced developer, so I am very happy to receive your contributions to improve the code quality and features of the framework. Please feel free to open an issue or pull request to contribute to the project 🥰.

## Acknowledgement
Special thanks to [Huggingface's Accelerate](https://github.com/huggingface/accelerate) and [Pytorch Lightning](https://github.com/Lightning-AI/lightning) for providing the great frameworks for training deep learning models.
