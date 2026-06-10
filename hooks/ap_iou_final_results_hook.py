"""
This hook gives the final AP and IoU numbers on the 
testing dataset 
"""

import os
import json
from typing import Dict
import matplotlib.pyplot as plt
import torch
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.events import get_event_storage

from .iou_evaluator import PerImageIoUEvaluator

class AP_IOU_FinalResults(HookBase): 
    def __init__(self, output_dir: str, cfg):
        super().__init__()
        self.cfg = cfg
        self.output_dir = output_dir

        # this is specific to this projects config file 
        self.model_name = getattr(cfg, "MODELNAME", getattr(cfg, "JOBNAME", "model"))

        os.makedirs(output_dir, exist_ok=True)
    
    def after_train(self):
        # set output dir and create 
        json_out = os.path.join(self.output_dir, f"json_{self.model_name}")
        os.makedirs(json_out, exist_ok=True)

        AP_dict = self._create_ap_dict()

        json_file_path = os.path.join(json_out, "ap_results.json")
        with open(json_file_path, "w") as f:
            json.dump(AP_dict, f, indent=4)

        # Save the IoU results as well in a separate file
        iou_results = self._get_IoU_numbers()
        iou_file_path = os.path.join(json_out, "iou_results.json")
        with open(iou_file_path, "w") as f:
            json.dump(iou_results, f, indent=4)

    def _get_IoU_numbers(self): 
        # gets the IoU numbers from the evaluator 
        if not self.cfg.DATASETS.TEST:
            return {}
        evaluator = PerImageIoUEvaluator(self.cfg.DATASETS.TEST[0])
        val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])

        results = inference_on_dataset(self.trainer.model, val_loader, evaluator)
        return results
        
    def _create_ap_dict(self) -> Dict: 
        # get the ap numbers 
        metrics = self._get_ap_numbers()
        
        # metrics is already a dictionary returned by _get_ap_numbers
        return metrics

    def _get_ap_numbers(self) -> Dict:
        cfg = self.trainer.cfg
        if not cfg.DATASETS.TEST:
            return {}
        evaluator = COCOEvaluator(
            cfg.DATASETS.TEST[0],
            distributed=(cfg.MODEL.DEVICE != "cpu"),
            output_dir=cfg.OUTPUT_DIR,
        )
        val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        results = inference_on_dataset(self.trainer.model, val_loader, evaluator)

        # retrieve all segmentation metrics
        segm_res = results.get("segm", {})
        
        metrics = {
             "AP":    segm_res.get("AP",    0.0), # IoU=0.50:0.95
             "AP50":  segm_res.get("AP50",  0.0), # IoU=0.50
             "AP75":  segm_res.get("AP75",  0.0), # IoU=0.75
             "APs":   segm_res.get("APs",   0.0), # Small objects
             "APm":   segm_res.get("APm",   0.0), # Medium objects
             "APl":   segm_res.get("APl",   0.0)  # Large objects
        }
        
        print(f"[Results] AP: {metrics['AP']:.3f}, AP50: {metrics['AP50']:.3f}, AP75: {metrics['AP75']:.3f}")
        return metrics
