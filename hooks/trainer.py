from detectron2.engine import DefaultPredictor, DefaultTrainer

from hooks.IoU_val_hook import IoUHook
from hooks.AP_val_hook import APVisualizationHook
from hooks.loss_hook import TrainingLossHook


class Trainer(DefaultTrainer):
    """Custom trainer subclass for SensAug. Register custom hooks here."""

    def build_hooks(self):
        hooks = super().build_hooks()
        eval_period = getattr(self.cfg, "EVAL_PERIOD", 100)
        hooks.append(
            IoUHook(
                output_dir=getattr(self.cfg, "OUTPUT_DIR", ""),
                job_name=getattr(self.cfg, "JOBNAME", ""),
                eval_period=eval_period,
            ),
            APVisualizationHook(
                output_dir=getattr(self.cfg, "OUTPUT_DIR", ""),
                eval_period=eval_period,
                job_name=getattr(self.cfg, "JOBNAME", ""),
            ),
            TrainingLossHook(
                output_dir=getattr(self.cfg, "OUTPUT_DIR", ""),
                job_name=getattr(self.cfg, "JOBNAME", ""),
            ),
        )
        return hooks
