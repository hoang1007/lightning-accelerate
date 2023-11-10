from __future__ import annotations
from warnings import warn
from typing import TYPE_CHECKING, Dict, Sequence, Union
from typing import List, Iterable

import torch
from lightning_accelerate.hooks import BaseHook
from lightning_accelerate.utils.config_utils import ConfigMixin

if TYPE_CHECKING:
    from lightning_accelerate import Trainer
    from lightning_accelerate.utils.progess_bar import BaseProgessBar


class TrainingModule(torch.nn.Module, BaseHook, ConfigMixin):
    LORA_TARGET_MODULES = None

    def training_step(self, batch, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        """
        Args:
            batch: The current batch.
            batch_idx: The index of the current batch.
            optimizer_idx: The index of the current optimizer.

        Returns:
            Tensor containing the loss for the current step.
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int):
        """
        Args:
            batch: The current batch.
            batch_idx: The index of the current batch.
        """
        raise NotImplementedError

    def get_optim_params(
        self,
    ) -> Union[
        Iterable[torch.nn.Parameter],
        Sequence[Iterable[torch.nn.Parameter]],
        Iterable[Dict],
        Sequence[Iterable[Dict]],
    ]:
        """
        The parameters to optimize.

        Returns:
            Parameter groups to be optimize.
            They could be an `Iterable[torch.nn.Parameter]` for single optimizer,
            `Sequence[Iterable[torch.nn.Parameter]]` for multiple optimizers,
            `Iterable[Dict]` defining param groups for advanced options. Dict should contain a `params` key, containing an iterable of parameters
            , `lr_scale` key containing a scale factor to apply to the learning rate
            and other optional optimizer arguments (See `torch.optim.Optimizer` for details)
            or `Sequence[Iterable[Dict]]` for multiple optimizers.
        """
        raise NotImplementedError

    @property
    def trainer(self):
        """
        The trainer object.
        """
        return self._trainer if hasattr(self, "_trainer") else None

    @property
    def progress_bar(self):
        """
        Progress bar for the current epoch.
        """
        return self._progess_bar if hasattr(self, "_progess_bar") else None

    @property
    def global_step(self):
        """
        The current global step.
        """
        return self.trainer.global_step if self.trainer is not None else 0

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    @property
    def logged_values(self):
        """
        The values logged in the current step.
        """
        return getattr(self, "_log_values", {})

    def log(self, values: dict, logger: bool = True, progess_bar: bool = True):
        """
        Log metrics for the current step to the logger and progess bar.

        Args:
            values: Dictionary of metrics to log.
            logger: Whether to log to the logger.
            progess_bar: Whether to log to the progess bar.
        """

        if hasattr(self, "_log_values"):
            self._log_values.update(values)
        else:
            self._log_values = values.copy()

        if self.trainer is None:
            warn("No trainer is registered to the training module!")
            return

        if self.trainer.accelerator.is_main_process:
            if progess_bar:
                if self.progress_bar is not None:
                    self.progress_bar.log(values)
                else:
                    warn("No progess bar found. Skipping progess bar logging.")
            if logger:
                self.trainer.accelerator.log(values, step=self.global_step)

    def log_images(self, images: dict):
        """
        Log images for the current step to the logger.

        Args:
            images: Dictionary of images to log.
        """
        if self.trainer is None:
            warn("No trainer is registered to the training module!")
            return

        if self.trainer.accelerator.is_main_process:
            tracker = self.trainer.get_tracker()
            if tracker is None:
                warn("No tracker found. Skipping image logging.")
            elif hasattr(tracker, "log_images"):
                tracker.log_images(images, step=self.global_step)
            else:
                warn(
                    f"Tracker {tracker.__class__.__name__} does not support image logging. Skipping..."
                )

    def register_trainer(self, trainer: Trainer):
        self._trainer = trainer

    def register_progress_bar(self, progress_bar: BaseProgessBar):
        self._progess_bar = progress_bar

    def save_pretrained(self, output_dir: str):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded.
        """
