import json
import os

import matplotlib.pyplot as plt
import torch
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage


class TrainingLossHook(HookBase):
    """Detectron2 hook that records and visualizes training and validation losses.

    Tracks ``total_loss`` and ``loss_mask`` from the Detectron2 event storage at
    every training step. If a validation dataloader is provided, validation losses
    are computed every ``eval_period`` iterations by running a forward pass in
    train mode (required to obtain the loss dict from Detectron2 models).

    After training, saves loss curve plots as PNG and optionally exports the raw
    loss values to JSON.

    Note:
        This hook is specific to instance segmentation models that report
        ``loss_mask`` (e.g. Mask R-CNN). It will error if ``loss_mask`` is not
        present in the event storage.
    """

    def __init__(self, output_dir, job_name, val_loss_loader=None, save_data=True):
        """Initialize the training loss hook.

        Args:
            output_dir: Directory to save loss plots and JSON data.
                Created automatically if it does not exist.
            job_name: Job name used in plot titles and output filenames.
            val_loss_loader: Optional dataloader for computing validation loss.
                If None, only training losses are tracked.
            save_data: If True, export raw loss values as JSON after training.
        """
        self.save_data = save_data
        self.output_dir = output_dir
        self.eval_period = 50
        self.loss_dict_total_train = {}
        self.loss_dict_mask_train = {}
        self.loss_dict_total_val = {}
        self.loss_dict_mask_val = {}

        self.val_loss_loader = val_loss_loader
        self.job_name = job_name

        os.makedirs(output_dir, exist_ok=True)

    def after_step(self):
        """Record training losses at every step; compute validation losses periodically."""
        storage = get_event_storage()
        iteration = self.trainer.iter

        current_scalars = storage.latest_with_smoothing_hint()
        total_loss = current_scalars["total_loss"][0]
        mask_loss = current_scalars["loss_mask"][0]

        self.loss_dict_total_train[iteration] = total_loss
        self.loss_dict_mask_train[iteration] = mask_loss

        if self.val_loss_loader is not None and iteration % self.eval_period == 0 and iteration != 0:
            val_total, val_mask = self._get_val_loss()
            self.loss_dict_total_val[iteration] = val_total
            self.loss_dict_mask_val[iteration] = val_mask

    @torch.no_grad()
    def _get_val_loss(self):
        """Compute average total and mask loss over the validation dataloader.

        The model is kept in train mode (required for Detectron2 models to
        return a loss dict) but gradients are disabled via ``@torch.no_grad()``.

        Returns:
            Tuple of (mean_total_loss, mean_mask_loss) over all batches.
        """
        model = self.trainer.model
        model.train()  # must be in train mode to get loss_dict from D2 models
        total_loss, mask_loss, n = 0.0, 0.0, 0
        for batch in self.val_loss_loader:
            loss_dict = model(batch)
            total_loss += sum(loss_dict.values()).item()
            mask_loss += loss_dict.get("loss_mask", torch.tensor(0.0)).item()
            n += 1
        return total_loss / max(n, 1), mask_loss / max(n, 1)

    def after_train(self):
        """Save loss plots and optionally export raw loss data as JSON."""
        print("Training Completed, now saving plots....")
        self._save_plots()
        if self.save_data:
            self._save_data()

    def _save_plots(self):
        """Save a 2-panel figure with total loss and mask loss curves."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Losses for {self.job_name} — {self.trainer.max_iter} iterations",
            fontsize=14,
        )

        axs[0].plot(
            list(self.loss_dict_total_train.keys()),
            list(self.loss_dict_total_train.values()),
            label="train",
        )
        if self.loss_dict_total_val:
            axs[0].plot(
                list(self.loss_dict_total_val.keys()),
                list(self.loss_dict_total_val.values()),
                label="val",
            )
        axs[0].set_title("Total Loss")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(
            list(self.loss_dict_mask_train.keys()),
            list(self.loss_dict_mask_train.values()),
            label="train",
        )
        if self.loss_dict_mask_val:
            axs[1].plot(
                list(self.loss_dict_mask_val.keys()),
                list(self.loss_dict_mask_val.values()),
                label="val",
            )
        axs[1].set_title("Mask Loss")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Loss")
        axs[1].legend()
        axs[1].grid(True)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(self.output_dir, f"{self.job_name}_loss_plot.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved loss curves to {out_path}")

    def _save_data(self):
        """Export all collected loss values to a JSON file."""
        data = {
            "train_total_loss": {str(k): v for k, v in self.loss_dict_total_train.items()},
            "train_mask_loss":  {str(k): v for k, v in self.loss_dict_mask_train.items()},
            "val_total_loss":   {str(k): v for k, v in self.loss_dict_total_val.items()},
            "val_mask_loss":    {str(k): v for k, v in self.loss_dict_mask_val.items()},
        }
        out_path = os.path.join(self.output_dir, f"{self.job_name}_loss_data.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved loss data to {out_path}")