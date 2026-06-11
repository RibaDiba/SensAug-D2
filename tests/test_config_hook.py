import os
import tempfile
from detectron2.config import get_cfg
from hooks.config_hook import TrainingConfigHook

class DummyTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

def test_training_config_hook():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a detectron2 CfgNode
        cfg = get_cfg()
        cfg.SOLVER.MAX_ITER = 42
        cfg.SOLVER.BASE_LR = 0.01
        
        # Instantiate the hook and mock trainer
        hook = TrainingConfigHook(output_dir=tmp_dir)
        trainer = DummyTrainer(cfg)
        hook.trainer = trainer
        
        # Trigger the hook
        hook.before_train()
        
        # Check if file exists
        expected_path = os.path.join(tmp_dir, "training_config.yaml")
        assert os.path.exists(expected_path)
        
        # Read the file and verify it matches the dumped config
        with open(expected_path, "r") as f:
            saved_content = f.read()
            
        assert "MAX_ITER: 42" in saved_content
        assert "BASE_LR: 0.01" in saved_content
