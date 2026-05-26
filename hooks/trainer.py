from detectron2.engine import DefaultPredictor, DefaultTrainer

class Trainer(DefaultTrainer):
    """
    Custom trainer subclass for SensAug. Register custom hooks here.
    """

    def build_hooks(self):
        hooks = super().build_hooks()
        return hooks