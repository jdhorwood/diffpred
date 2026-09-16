# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union

import lightning.pytorch
from lightning.pytorch.callbacks.model_checkpoint import (
    ModelCheckpoint as PTLModelCheckpoint,
)
from lightning.pytorch.utilities import rank_zero_info
from diffpred.callbacks.ema import EMA


class ModelCheckpoint(PTLModelCheckpoint):
    """Light wrapper around Lightning's ModelCheckpoint to force a saved checkpoint on train_end.
    Adds support for asyncronous checkpointing and provides some additional logic to clean up invalid checkpoints
    Args:
        monitor: Metric to monitor when saving top-k checkpoints.
        verbose: Verbosity mode.
        save_last: When ``True``, saves a `*-last` copy whenever a checkpoint file gets saved.
        save_top_k: When ``True``, saves the top-k checkpoints according to ``monitor``.
        save_weights_only:  if ``True``, then only the model's weights will be saved. Optimizer states will
            be omitted from all checkpoints.
        mode: One of {min, max}. Whether the objective is to minimize or maximize the monitored quantity.
        every_n_epochs: Number of epochs between checkpoints.
        every_n_train_steps: Number of train steps between checkpoints.
        train_time_interval: After each interval, monitor checkpoints. Not to be used with
            ``every_n_epochs`` or ``every_n_train_steps``.
        save_on_train_epoch_end: Whether to run checkpointing at the end of the training epoch
        save_optim_on_train_end: Whether to include the optimizer states in the final checkpoint
            at the end of training. Only applicable when save_weights_only is ``False``.
        always_save_context: Whether to dump the artifacts needed to reinintialize the current
            model, trainer, and dataloader to allow for reproducibility of experiments.
        save_context_on_train_end: Whether to dump the artifacts on_train_end regardless of whether
            ``always_save_context`` is ``True``.
        async_save: Whether to enable asynchronous checkpointing.
    """

    UNFINISHED_CHECKPOINT_SUFFIX = "-unfinished"

    def __init__(
        self,
        monitor: Optional[str] = "val_loss",
        verbose: bool = False,
        save_last: Optional[Union[bool, Literal["link"]]] = True,
        save_top_k: int = 3,
        save_weights_only: bool = False,
        mode: str = "min",
        every_n_epochs: int = 1,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        save_on_train_epoch_end: Optional[
            bool
        ] = False,  # Save after training, not after validation
        save_optim_on_train_end: Optional[bool] = False,
        always_save_context: bool = True,
        save_context_on_train_end: bool = True,
        **kwargs,
    ):
        self.always_save_context = always_save_context
        self.save_context_on_train_end = save_context_on_train_end
        self.save_optim_on_train_end = save_optim_on_train_end

        ## stores the next -last checkpoint to be saved, used only when save_last = 'link'
        ## this is needed because when using symlinks, we need to update the non-last checkpoint's
        ## last_model_path to point to the corresponding -last version
        self.future_last_model_path = ""

        # Checkpoints which removal is deferred until async save is done.
        # Each element of `deferred_ckpts_to_remove` is a growing list
        # that `self._remove_checkpoint` adds to. Once `self._save_checkpoint`
        # is called, the last element is frozen and a new element is added.
        self.deferred_ckpts_to_remove: List[List[str]] = []
        self.ckpts_to_link: Dict[str, str] = {}

        # Call the parent class constructor with the remaining kwargs.
        super().__init__(
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            every_n_epochs=every_n_epochs,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            save_on_train_epoch_end=save_on_train_epoch_end,
            **kwargs,
        )

    def _ema_callback(self, trainer: "lightning.pytorch.Trainer"):
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    def _save_checkpoint(
        self, trainer: "lightning.pytorch.Trainer", filepath: str
    ) -> None:
        ema_callback = self._ema_callback(trainer)

        self._last_global_step_saved = trainer.global_step

        if ema_callback is not None:
            with ema_callback.save_original_optimizer_state(trainer):
                super()._save_checkpoint(trainer, filepath)

            # save EMA copy of the model as well.
            with ema_callback.save_ema_model(trainer):
                rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
                filepath = self._ema_format_filepath(filepath)
                if self.verbose:
                    rank_zero_info(
                        f"Saving EMA weights to separate checkpoint {filepath}"
                    )
                super()._save_checkpoint(trainer, filepath)
        else:
            super()._save_checkpoint(trainer, filepath)

    def _remove_checkpoint(
        self, trainer: "lightning.pytorch.Trainer", filepath: str
    ) -> None:
        """Performs checkpoint removal."""

        super()._remove_checkpoint(trainer, filepath)
        ema_callback = self._ema_callback(trainer)
        if ema_callback is not None:
            # remove EMA copy of the state dict as well.

            filepath = self._ema_format_filepath(filepath)
            super()._remove_checkpoint(trainer, filepath)

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}")

    def _has_ema_ckpts(self, checkpoints: Iterable[Path]) -> bool:
        return any(
            self._is_ema_filepath(checkpoint_path) for checkpoint_path in checkpoints
        )

    def _is_ema_filepath(self, filepath: Union[Path, str]) -> bool:
        return str(filepath).endswith(f"-EMA{self.FILE_EXTENSION}")

    @property
    def _saved_checkpoint_paths(self) -> Iterable[Path]:
        # distributed checkpoints are directories so we check for them here
        # we filter out unfinished checkpoints, these should be deleted during next cleanup
        dist_checkpoints = [d for d in Path(self.dirpath).glob("*") if d.is_dir()]
        if dist_checkpoints:
            return filter(
                lambda p: not self.is_checkpoint_unfinished(p), dist_checkpoints
            )
        else:
            checkpoint_files = [f for f in Path(self.dirpath).rglob("*.ckpt")]
            return filter(
                lambda p: not self.is_checkpoint_unfinished(p), checkpoint_files
            )
