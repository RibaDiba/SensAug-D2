import os, json
import matplotlib.pyplot as plt
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.events import get_event_storage


class APVisualizationHook(HookBase):
    """
    A Detectron2 training hook that periodically evaluates instance segmentation
    AP metrics on the validation set and saves a plot + JSON of the results.

    Evaluation runs every `eval_period` iterations (excluding iteration 0).
    At the end of training, a matplotlib figure is saved and optionally the
    raw per-iteration numbers are serialized to JSON.

    The hook uses `cfg.DATASETS.TEST[1]` as the validation split — index 1
    assumes the dataset config lists [train, val] under DATASETS.TEST.
    """

    def __init__(self, output_dir: str, eval_period: int = 100, save_data: bool = True, job_name: str = ""):
        """
        Args:
            output_dir: Directory where the AP plot and JSON results are written.
                        Created automatically if it does not exist.
            cfg: Detectron2 CfgNode used to build the evaluator and data loader.
            eval_period: Number of iterations between validation evaluations.
            save_data: If True, serialize the collected AP values to a JSON file
                       after training completes.
            job_name: Job name used in plot titles and filenames.
        """
        super().__init__()
        self.output_dir = output_dir
        self.eval_period = eval_period
        self.save_data = save_data

        self.job_name = job_name

        self.AP_dict = {}
        self.AP_50 = {}
        self.AP_75 = {}
        os.makedirs(output_dir, exist_ok=True)

    def after_step(self):
        """Evaluate AP metrics every `eval_period` iterations."""
        storage = get_event_storage()
        iteration = storage.iter if hasattr(storage, "iter") else storage.iteration

        if iteration % self.eval_period == 0 and iteration != 0:
            self.AP_dict[iteration], self.AP_75[iteration], self.AP_50[iteration] = (
                self._get_ap_numbers()
            )

    def after_train(self):
        """Save the AP curve plot (and optionally JSON data) after training ends."""
        plt.plot(list(self.AP_dict.keys()), list(self.AP_dict.values()), label="AP")
        plt.plot(list(self.AP_75.keys()), list(self.AP_75.values()), label="AP75")
        plt.plot(list(self.AP_50.keys()), list(self.AP_50.values()), label="AP50")
        plt.title(f"AP Curve - {self.job_name} - Validation Set")
        plt.xlabel("Iteration")
        plt.ylabel("AP")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.output_dir, f"{self.job_name}_AP_plot.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved AP curve to {save_path}")

        if self.save_data:
            self._save_data()

    def _get_ap_numbers(self) -> tuple[float, float, float]:
        """
        Run inference on the validation set and return (mAP, AP75, AP50).

        Uses `cfg.DATASETS.TEST[1]` as the validation dataset name and pulls
        segmentation metrics from the COCO evaluator results.
        """
        cfg = self.trainer.cfg
        val_dataset = cfg.DATASETS.TEST[1]

        evaluator = COCOEvaluator(
            val_dataset,
            distributed=(cfg.MODEL.DEVICE != "cpu"),
            output_dir=cfg.OUTPUT_DIR,
        )
        val_loader = build_detection_test_loader(cfg, val_dataset)
        results = inference_on_dataset(self.trainer.model, val_loader, evaluator)

        segm = results.get("segm", {})
        mAP = segm.get("AP", 0.0)
        mAP75 = segm.get("AP75", 0.0)
        mAP50 = segm.get("AP50", 0.0)
        print(f"[Iter {self.trainer.iter} AP] → mAP@[.5:.95] = {mAP:.3f}")
        return mAP, mAP75, mAP50

    def _save_data(self):
        """Serialize the collected AP dicts to JSON under `output_dir/json_<run_name>/results.json`."""
        json_out = os.path.join(self.output_dir, f"json_{self.job_name}")
        os.makedirs(json_out, exist_ok=True)

        full_dict = {"AP": self.AP_dict, "AP75": self.AP_75, "AP50": self.AP_50}

        json_file_path = os.path.join(json_out, "results.json")
        with open(json_file_path, "w") as f:
            json.dump(full_dict, f, indent=4)