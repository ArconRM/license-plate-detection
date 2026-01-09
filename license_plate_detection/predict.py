from ultralytics import YOLO


import os
import json
import glob
from pathlib import Path
from typing import List
import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


def predict_yolo(model_path, source, save_name, batch_size=1, device="mps"):
    """
    Run YOLO prediction on a given source with specified parameters.

    Args:
      model_path (str): Path to the trained YOLO model weights.
      source (str): Path or source to the images or video for prediction.
      save_name (str): Name for saving prediction results.
      batch_size (int): Number of images to process per batch.
      device (str): Device to run inference on, e.g., 'cpu', 'cuda', 'mps'.
    """
    model = YOLO(model_path)

    results = model.predict(
        source=source,
        batch=batch_size,
        save=True,
        name=save_name,
        device=device
    )
    return results


class FixedColorVisualizer(Visualizer):
    def _jitter(self, color):
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

    save_dir = "../runs/detect/detectron2"
    os.makedirs(save_dir, exist_ok=True)

    for img_path in image_paths:
        im = cv2.imread(img_path)
        outputs = predictor(im)

        v = FixedColorVisualizer(im[:, :, ::-1], metadata=metadata, scale=2.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])

        # plt.figure(figsize=(20, 10))
        # plt.imshow(out.get_image())
        # plt.axis("off")
        # plt.show()


def inference_detectron(
        raw_data_dir: str,
        crops_data_dir: str,
        model_weights: str,
        config_yaml: str,
        model_type: str = "faster_rcnn",  # "faster_rcnn" | "retinanet"
        num_classes: int = 1,
        score_threshold: float = 0.5,
        input_size: int = 640,
        padding: int = 30,
        device: str = "cpu",
        topk_candidates: int = 300,  # только для RetinaNet
        image_extensions: List[str] = None,
):
    """
    Universal Detectron2 inference function (FasterRCNN / RetinaNet)

    model_type:
        - "faster_rcnn"
        - "retinanet"
    """

    if image_extensions is None:
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    # ----------------------------
    # Config
    # ----------------------------
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_yaml))

    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = device

    cfg.INPUT.MIN_SIZE_TEST = input_size
    cfg.INPUT.MAX_SIZE_TEST = 1000

    if model_type == "retinanet":
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = topk_candidates
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold

        # CPU-friendly RPN
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500

    predictor = DefaultPredictor(cfg)

    # ----------------------------
    # Collect images
    # ----------------------------
    frame_paths = []
    for ext in image_extensions:
        frame_paths.extend(
            glob.glob(os.path.join(raw_data_dir, "**", ext), recursive=True)
        )
    frame_paths = sorted(frame_paths)

    print(f"Found {len(frame_paths)} frames in {raw_data_dir}")

    if not frame_paths:
        return {"status": "error", "message": "No images found"}

    os.makedirs(crops_data_dir, exist_ok=True)

    all_detections = []
    crop_index = 0

    # ----------------------------
    # Inference loop
    # ----------------------------
    for idx, fp in enumerate(frame_paths):
        if (idx + 1) % 10 == 0:
            print(f"Processing image {idx + 1}/{len(frame_paths)}")

        img = cv2.imread(fp)
        if img is None:
            print(f"Warning: Could not read image {fp}")
            continue

        h, w = img.shape[:2]
        time_label = Path(fp).stem.split("_")[-1]

        import time
        t0 = time.time()
        outputs = predictor(img)
        inference_ms = (time.time() - t0) * 1000

        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            if x1 >= x2 or y1 >= y2:
                continue

            cx1 = max(0, x1 - padding)
            cy1 = max(0, y1 - padding)
            cx2 = min(w, x2 + padding)
            cy2 = min(h, y2 + padding)

            if cx1 >= cx2 or cy1 >= cy2:
                continue

            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            crop_filename = f"crop_{time_label}_{crop_index}.png"
            crop_path = os.path.join(crops_data_dir, crop_filename)

            if not cv2.imwrite(crop_path, crop):
                continue

            all_detections.append({
                "frame": fp,
                "crop_index": crop_index,
                "crop_filename": crop_filename,
                "coordinates": {
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "crop_x1": cx1,
                    "crop_y1": cy1,
                    "crop_x2": cx2,
                    "crop_y2": cy2
                },
                "confidence": float(score),
                "class": int(cls),
                "speed": {"inference_ms": round(inference_ms, 2)}
            })

            crop_index += 1

    # ----------------------------
    # Save JSON
    # ----------------------------
    result_file = os.path.join(
        crops_data_dir,
        f"detectron2_{model_type}_results.json"
    )

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)

    print(f"Finished. {crop_index} crops saved")

    return {
        "status": "ok",
        "model_type": model_type,
        "crops_saved": crop_index,
        "total_frames": len(frame_paths),
        "result_file": result_file
    }
