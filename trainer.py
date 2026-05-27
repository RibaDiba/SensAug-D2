from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader

from augmentations import build_augmentations
from hooks.IoU_val_hook import IoUHook
from hooks.AP_val_hook import APVisualizationHook
from hooks.loss_hook import TrainingLossHook


class Trainer(DefaultTrainer):
    """Custom trainer subclass for SensAug. Register custom hooks here."""

    @classmethod
    def build_train_loader(cls, cfg):
        augs = build_augmentations()
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        eval_period = getattr(self.cfg, "EVAL_PERIOD", 100)
        hooks.append(
            IoUHook(
                output_dir=getattr(self.cfg, "OUTPUT_DIR", ""),
                job_name=getattr(self.cfg, "JOBNAME", ""),
                eval_period=eval_period,
            ),
        )
        hooks.append(
            APVisualizationHook(
                output_dir=getattr(self.cfg, "OUTPUT_DIR", ""),
                eval_period=eval_period,
                job_name=getattr(self.cfg, "JOBNAME", ""),
            ),
        )
        hooks.append(
            TrainingLossHook(
                output_dir=getattr(self.cfg, "OUTPUT_DIR", ""),
                job_name=getattr(self.cfg, "JOBNAME", ""),
            ),
        )
        return hooks
