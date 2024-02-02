"""Base class used for all pytorch lightning modules."""
import collections
import logging
import warnings
from typing import Any

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor, nn
from torch_geometric.data import Data

from gnn_tracking.utils.lightning import StandardError, obj_from_or_to_hparams
from gnn_tracking.utils.log import get_logger
from gnn_tracking.utils.oom import tolerate_some_oom_errors

# The following abbreviations are used throughout the code:
# W: edge weights
# B: condensation likelihoods
# H: clustering coordinates
# Y: edge truth labels
# L: hit truth labels
# P: Track parameters


class ImprovedLogLM(LightningModule):
    def __init__(self, **kwargs):
        """This subclass of `LightningModule` adds some convenience to logging,
        e.g., logging of statistical uncertainties (batch-to-batch) and logging
        of the validation metrics to the console after each validation epoch.
        """
        super().__init__(**kwargs)
        self._uncertainties = collections.defaultdict(StandardError)
        self.print_validation_results = True

    def log_dict_with_errors(self, dct: dict[str, float], batch_size=None) -> None:
        """Log a dictionary of values with their statistical uncertainties.

        This method only starts calculating the uncertainties. To log them,
        `_log_errors` needs to be called at the end of the train/val/test epoch
        (done with the hooks configured in this class).
        """
        self.log_dict(
            dct,
            on_epoch=True,
            batch_size=batch_size,
        )
        for k, v in dct.items():
            if f"{k}_std" in dct or k.endswith("_std"):
                continue
            self._uncertainties[k](torch.Tensor([v]))

    def _log_errors(self) -> None:
        """Log the uncertainties calculated in `log_dict_with_errors`.
        Needs to be called at the end of the train/val/test epoch.
        """
        ...
        for k, v in self._uncertainties.items():
            self.log(k + "_std", v.compute(), on_epoch=True, batch_size=1)
        self._uncertainties.clear()

    # noinspection PyUnusedLocal
    def on_train_epoch_end(self, *args) -> None:
        self._log_errors()

    def on_validation_epoch_end(self) -> None:
        self._log_errors()

    def on_test_epoch_end(self) -> None:
        self._log_errors()


class TrackingModule(ImprovedLogLM):
    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        preproc: nn.Module | None = None,
    ):
        """Base class for all pytorch lightning modules in this project."""
        super().__init__()
        self.model = obj_from_or_to_hparams(self, "model", model)
        self.logg = get_logger("TM", level=logging.DEBUG)
        self.preproc = obj_from_or_to_hparams(self, "preproc", preproc)
        self.optimizer = optimizer
        self.scheduler = scheduler

        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated. It will be removed in the future and "
            "UntypedStorage will be the only storage class.",
        )

    def forward(self, data: Data, _preprocessed=False) -> Tensor | dict[str, Tensor]:
        if not _preprocessed:
            data = self.data_preproc(data)
        return self.model.forward(data)

    def data_preproc(self, data) -> Data:
        if self.preproc is not None:
            return self.preproc(data)
        return data

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    @tolerate_some_oom_errors
    def backward(self, *args: Any, **kwargs: Any) -> None:
        super().backward(*args, **kwargs)
