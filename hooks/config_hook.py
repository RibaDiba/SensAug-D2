import os
from detectron2.engine.hooks import HookBase

class TrainingConfigHook(HookBase):
    """A Detectron2 training hook that saves the training configuration.

    At the beginning of training (in ``before_train``), it serializes and
    writes the configuration node (``self.trainer.cfg``) to the model's
    output directory as ``training_config.yaml``.
    """

    def __init__(self, output_dir: str):
        """Initialize the training config hook.

        Args:
            output_dir: Directory where the training config is saved.
                        Created automatically if it does not exist.
        """
        super().__init__()
        self.output_dir = output_dir

    def before_train(self):
        """Save the training configuration when training starts."""
        if not self.output_dir:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        config_path = os.path.join(self.output_dir, "training_config.yaml")
        
        # self.trainer.cfg is a detectron2.config.CfgNode
        # which inherits from yacs.config.CfgNode and has a dump() method
        cfg = self.trainer.cfg
        with open(config_path, "w") as f:
            f.write(cfg.dump())
            
        print(f"[TrainingConfigHook] Saved training config to {config_path}")
