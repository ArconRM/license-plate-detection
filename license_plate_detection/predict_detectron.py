import os
import glob
import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo


class FixedColorVisualizer(Visualizer):
    def _jitter(self, color):
        # Disable jittering by returning color as-is
        return color


def predict_detectron(
        model_weights: str,
        num_classes: int,
        test_data_dir: str,
        dataset_name: str,
        score_thresh: float = 0.5,
        device: str = "cpu",
        config_yaml: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
):
    """
    Run object detection on all images in a directory using Detectron2 and visualize results.

    Args:
      model_weights (str): Path to the trained model weights file.
      num_classes (int): Number of classes the model predicts.
      test_data_dir (str): Directory containing test images.
      dataset_name (str): Registered Detectron2 dataset name for metadata.
      score_thresh (float): Threshold for prediction confidence score.
      device (str): Device to run inference on, e.g., 'cpu', 'cuda'.
      config_yaml (str): Detectron2 config file path or model zoo config name.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_yaml))
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = device

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(dataset_name)

    image_paths = glob.glob(os.path.join(test_data_dir, "*.*"))

    for img_path in image_paths:
        im = cv2.imread(img_path)
        outputs = predictor(im)

        v = FixedColorVisualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=(20, 10))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.axis("off")
        plt.show()
