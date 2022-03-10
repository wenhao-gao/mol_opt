import sys

from pytorch_lightning.callbacks import ProgressBarBase
from tqdm import tqdm


class EpochAndStepProgressBar(ProgressBarBase):
    def __init__(self, refresh_rate: int = 100):
        super().__init__()
        self.step_bar = None
        self.epoch_bar = None
        self.refresh_rate = refresh_rate
        self.sanity_checking = False

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.epoch_bar = tqdm(
            desc="Training",
            unit="epoch",
            leave=True,
            dynamic_ncols=True,
            total=trainer.max_epochs,
        )
        self.step_bar = tqdm(
            desc="Epoch (train)",
            unit="step",
            leave=False,
            dynamic_ncols=True,
            total=self.total_train_batches + self.total_val_batches,
        )

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)

        self.step_bar.reset()
        self.step_bar.set_description_str("Epoch (train)")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )

        # loss = trainer.progress_bar_dict["loss"]
        loss = self.get_metrics(trainer, pl_module)["loss"]
        self.step_bar.set_postfix_str(f"loss={loss}")
        self.step_bar.update()

    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_epoch_start(trainer, pl_module)
        if self.sanity_checking:
            return

        self.step_bar.set_description_str("Epoch (validation)", True)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        if self.sanity_checking:
            return

        self.step_bar.update()

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.sanity_checking:
            return

        # NOTE(degraff): not sure why this try/except block is
        # necessary right now but for some reason the first training epoch
        # won't log the 'train_loss' key in trainer.callback_metrics
        try:
            train_loss = trainer.callback_metrics["train_loss"].item()
            val_loss = trainer.callback_metrics["val_loss"].item()
            self.epoch_bar.set_postfix_str(
                f"train_loss={train_loss:0.3f}, val_loss={val_loss:0.3f})",
                refresh=False,
            )
        except KeyError:
            pass

        self.epoch_bar.update()

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)

        self.epoch_bar.close()
        self.step_bar.close()

    def on_sanity_check_start(self, trainer, pl_module):
        super().on_sanity_check_start(trainer, pl_module)

        self.sanity_checking = True
        print("Sanity check ...", end=" ", file=sys.stderr)

    def on_sanity_check_end(self, trainer, pl_module):
        super().on_sanity_check_end(trainer, pl_module)

        self.sanity_checking = False
        print("Done!", file=sys.stderr, end="\r")
