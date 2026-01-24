from ultralytics import YOLO


import os
import json
import glob
from pathlib import Path
from typing import List
import cv2
from typing import List, Tuple
# from detectron2.data import MetadataCatalog
# from detectron2.utils.visualizer import Visualizer
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor

# import torch
# import numpy as np
# from pathlib import Path
# import onnxruntime as ort


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


# class FixedColorVisualizer(Visualizer):
#     def _jitter(self, color):
#         return color


# def predict_detectron(
#         model_weights: str,
#         num_classes: int,
#         test_data_dir: str,
#         dataset_name: str,
#         score_thresh: float = 0.5,
#         device: str = "cpu",
#         config_yaml: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# ):
#     """
#     Run object detection on all images in a directory using Detectron2 and visualize results.

#     Args:
#       model_weights (str): Path to the trained model weights file.
#       num_classes (int): Number of classes the model predicts.
#       test_data_dir (str): Directory containing test images.
#       dataset_name (str): Registered Detectron2 dataset name for metadata.
#       score_thresh (float): Threshold for prediction confidence score.
#       device (str): Device to run inference on, e.g., 'cpu', 'cuda'.
#       config_yaml (str): Detectron2 config file path or model zoo config name.
#     """
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(config_yaml))
#     cfg.MODEL.WEIGHTS = model_weights
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
#     cfg.MODEL.DEVICE = device

#     predictor = DefaultPredictor(cfg)
#     metadata = MetadataCatalog.get(dataset_name)

#     image_paths = glob.glob(os.path.join(test_data_dir, "*.*"))

#     save_dir = "../runs/detect/detectron2"
#     os.makedirs(save_dir, exist_ok=True)

#     for img_path in image_paths:
#         im = cv2.imread(img_path)
#         outputs = predictor(im)

#         v = FixedColorVisualizer(im[:, :, ::-1], metadata=metadata, scale=2.0)
#         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

#         save_path = os.path.join(save_dir, os.path.basename(img_path))
#         cv2.imwrite(save_path, out.get_image()[:, :, ::-1])

#         # plt.figure(figsize=(20, 10))
#         # plt.imshow(out.get_image())
#         # plt.axis("off")
#         # plt.show()


# def inference_detectron(
#         raw_data_dir: str,
#         crops_data_dir: str,
#         model_weights: str,
#         config_yaml: str,
#         model_type: str = "faster_rcnn",  # "faster_rcnn" | "retinanet"
#         num_classes: int = 1,
#         score_threshold: float = 0.5,
#         input_size: int = 640,
#         padding: int = 30,
#         device: str = "cpu",
#         topk_candidates: int = 300,  # только для RetinaNet
#         image_extensions: List[str] = None,
# ):
#     """
#     Universal Detectron2 inference function (FasterRCNN / RetinaNet)

#     model_type:
#         - "faster_rcnn"
#         - "retinanet"
#     """

#     if image_extensions is None:
#         image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

#     # ----------------------------
#     # Config
#     # ----------------------------
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(config_yaml))

#     cfg.MODEL.WEIGHTS = model_weights
#     cfg.MODEL.DEVICE = device

#     cfg.INPUT.MIN_SIZE_TEST = input_size
#     cfg.INPUT.MAX_SIZE_TEST = 1000

#     if model_type == "retinanet":
#         cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
#         cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_threshold
#         cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = topk_candidates
#     else:
#         cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
#         cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold

#         # CPU-friendly RPN
#         cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
#         cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500

#     predictor = DefaultPredictor(cfg)

#     # ----------------------------
#     # Collect images
#     # ----------------------------
#     frame_paths = []
#     for ext in image_extensions:
#         frame_paths.extend(
#             glob.glob(os.path.join(raw_data_dir, "**", ext), recursive=True)
#         )
#     frame_paths = sorted(frame_paths)

#     print(f"Found {len(frame_paths)} frames in {raw_data_dir}")

#     if not frame_paths:
#         return {"status": "error", "message": "No images found"}

#     os.makedirs(crops_data_dir, exist_ok=True)

#     all_detections = []
#     crop_index = 0

#     # ----------------------------
#     # Inference loop
#     # ----------------------------
#     for idx, fp in enumerate(frame_paths):
#         if (idx + 1) % 10 == 0:
#             print(f"Processing image {idx + 1}/{len(frame_paths)}")

#         img = cv2.imread(fp)
#         if img is None:
#             print(f"Warning: Could not read image {fp}")
#             continue

#         h, w = img.shape[:2]
#         time_label = Path(fp).stem.split("_")[-1]

#         import time
#         t0 = time.time()
#         outputs = predictor(img)
#         inference_ms = (time.time() - t0) * 1000

#         instances = outputs["instances"].to("cpu")
#         boxes = instances.pred_boxes.tensor.numpy()
#         scores = instances.scores.numpy()
#         classes = instances.pred_classes.numpy()

#         for box, score, cls in zip(boxes, scores, classes):
#             x1, y1, x2, y2 = map(int, box)
#             if x1 >= x2 or y1 >= y2:
#                 continue

#             cx1 = max(0, x1 - padding)
#             cy1 = max(0, y1 - padding)
#             cx2 = min(w, x2 + padding)
#             cy2 = min(h, y2 + padding)

#             if cx1 >= cx2 or cy1 >= cy2:
#                 continue

#             crop = img[cy1:cy2, cx1:cx2]
#             if crop.size == 0:
#                 continue

#             crop_filename = f"crop_{time_label}_{crop_index}.png"
#             crop_path = os.path.join(crops_data_dir, crop_filename)

#             if not cv2.imwrite(crop_path, crop):
#                 continue

#             all_detections.append({
#                 "frame": fp,
#                 "crop_index": crop_index,
#                 "crop_filename": crop_filename,
#                 "coordinates": {
#                     "x1": x1, "y1": y1,
#                     "x2": x2, "y2": y2,
#                     "width": x2 - x1,
#                     "height": y2 - y1,
#                     "crop_x1": cx1,
#                     "crop_y1": cy1,
#                     "crop_x2": cx2,
#                     "crop_y2": cy2
#                 },
#                 "confidence": float(score),
#                 "class": int(cls),
#                 "speed": {"inference_ms": round(inference_ms, 2)}
#             })

#             crop_index += 1

#     # ----------------------------
#     # Save JSON
#     # ----------------------------
#     result_file = os.path.join(
#         crops_data_dir,
#         f"detectron2_{model_type}_results.json"
#     )

#     with open(result_file, "w", encoding="utf-8") as f:
#         json.dump(all_detections, f, indent=2, ensure_ascii=False)

#     print(f"Finished. {crop_index} crops saved")

#     return {
#         "status": "ok",
#         "model_type": model_type,
#         "crops_saved": crop_index,
#         "total_frames": len(frame_paths),
#         "result_file": result_file
#     }


def inference_yolo(
        raw_data_dir: str,
        crops_data_dir: str,
        model_weights: str,
        score_threshold: float = 0.5,
        input_size: int = 640,
        padding: int = 30,
        device: str = "cpu",
        batch_size: int = 32,
        image_extensions: List[str] = None,
):
    """
    Universal YOLO inference function (v5 / v8 / v11 via Ultralytics)

    Args are intentionally aligned with inference_detectron
    """

    from ultralytics import YOLO
    import time
    import glob
    import json
    import os
    import cv2
    from pathlib import Path
    import torch

    if image_extensions is None:
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    # ----------------------------
    # Device
    # ----------------------------
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ----------------------------
    # Load model
    # ----------------------------
    model = YOLO(model_weights)
    model.fuse()

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
    start_time = time.time()

    # ----------------------------
    # Batch inference
    # ----------------------------
    for batch_start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[batch_start: batch_start + batch_size]
        images = []

        for fp in batch_paths:
            img = cv2.imread(fp)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Could not read image {fp}")

        if not images:
            continue

        t0 = time.time()
        results = model.predict(
            source=images,
            imgsz=input_size,
            conf=score_threshold,
            device=device,
            half=(device == "cuda"),
            verbose=False,
            batch=len(images),
        )
        inference_ms = (time.time() - t0) * 1000

        # ----------------------------
        # Process results
        # ----------------------------
        for fp, img, r in zip(batch_paths, images, results):
            h, w = img.shape[:2]
            time_label = Path(fp).stem.split("_")[-1]

            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

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
                    "confidence": float(box.conf.cpu().numpy()[0]),
                    "class": int(box.cls.cpu().numpy()[0]),
                    "speed": {
                        "inference_ms": round(inference_ms, 2)
                    }
                })

                crop_index += 1

    # ----------------------------
    # Save JSON
    # ----------------------------
    result_file = os.path.join(
        crops_data_dir,
        "yolo_results.json"
    )

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time

    print(f"Finished. {crop_index} crops saved")
    print(f"Elapsed time: {elapsed:.2f}s, FPS: {len(frame_paths) / elapsed:.2f}")

    return {
        "status": "ok",
        "crops_saved": crop_index,
        "total_frames": len(frame_paths),
        "result_file": result_file,
        "elapsed_time": elapsed
    }



def inference_yolo_torchscript(
        raw_data_dir: str,
        crops_data_dir: str,
        model_path: str,
        score_threshold: float = 0.5,
        input_size: int = 640,
        padding: int = 30,
        device: str = "cuda",
        image_extensions: List[str] = None,
):
    """
    YOLO TorchScript inference (for GPU)
    
    Args:
        raw_data_dir: Directory with input images
        crops_data_dir: Directory to save crops
        model_path: Path to .torchscript model
        score_threshold: Confidence threshold
        input_size: Model input size (640 for YOLOv11)
        padding: Padding around detected boxes for crops
        device: "cuda" or "cpu"
        image_extensions: List of image extensions to process
    """
    
    if image_extensions is None:
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    
    # ----------------------------
    # Load model
    # ----------------------------
    print(f"Loading TorchScript model from {model_path}")
    model = torch.jit.load(model_path)
    model = model.to(device)
    model.eval()
    
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
        
        # Preprocessing
        img_resized = cv2.resize(img, (input_size, input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Inference
        import time
        t0 = time.time()
        
        with torch.no_grad():
            predictions = model(img_tensor)
        # После predictions = model(img_tensor)
        print(f"Predictions type: {type(predictions)}")
        print(f"Predictions shape/len: {predictions[0].shape if hasattr(predictions[0], 'shape') else len(predictions)}")
        print(f"First prediction sample: {predictions[0][:3]}")  # Первые 3 элемента
        
        inference_ms = (time.time() - t0) * 1000
        
        # Parse predictions
        # YOLOv11 TorchScript output: [batch, num_boxes, 6]
        # где 6 = [x1, y1, x2, y2, confidence, class]
        # Координаты уже в xyxy формате и нормализованы к input_size
        
        pred = predictions[0].cpu()  # Берем batch 0
        
        # Фильтруем по confidence
        conf_mask = pred[:, 4] > score_threshold
        pred = pred[conf_mask]
        
        if len(pred) == 0:
            continue
        
        # Извлекаем данные
        boxes = pred[:, :4].numpy()
        scores = pred[:, 4].numpy()
        classes = pred[:, 5].numpy().astype(int)
        
        # Масштабируем координаты с input_size на оригинальный размер
        scale_x = w / input_size
        scale_y = h / input_size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        # Process detections
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Apply padding
            cx1 = max(0, x1 - padding)
            cy1 = max(0, y1 - padding)
            cx2 = min(w, x2 + padding)
            cy2 = min(h, y2 + padding)
            
            if cx1 >= cx2 or cy1 >= cy2:
                continue
            
            # Extract crop
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
    result_file = os.path.join(crops_data_dir, "yolo_torchscript_results.json")
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)
    
    print(f"Finished. {crop_index} crops saved")
    
    return {
        "status": "ok",
        "model_type": "yolo_torchscript",
        "crops_saved": crop_index,
        "total_frames": len(frame_paths),
        "result_file": result_file
    }


def inference_yolo_onnx(
        raw_data_dir: str,
        crops_data_dir: str,
        model_path: str,
        score_threshold: float = 0.5,
        input_size: int = 640,
        padding: int = 30,
        num_threads: int = 4,
        image_extensions: List[str] = None,
):
    """
    YOLO ONNX inference (for CPU)
    
    Args:
        raw_data_dir: Directory with input images
        crops_data_dir: Directory to save crops
        model_path: Path to .onnx model
        score_threshold: Confidence threshold
        input_size: Model input size (640 for YOLOv11)
        padding: Padding around detected boxes for crops
        num_threads: Number of CPU threads for inference
        image_extensions: List of image extensions to process
    """
    
    if image_extensions is None:
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    
    # ----------------------------
    # Load ONNX model
    # ----------------------------
    print(f"Loading ONNX model from {model_path}")
    
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = num_threads
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=['CPUExecutionProvider']
    )
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
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
        
        # Preprocessing
        img_resized = cv2.resize(img, (input_size, input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0)
        
        # Inference
        import time
        t0 = time.time()
        
        outputs = session.run(None, {input_name: img_batch})
        
        inference_ms = (time.time() - t0) * 1000
        
        # Parse ONNX output
        # YOLO ONNX обычно возвращает [1, num_predictions, 85] или [1, 25200, 85]
        # где 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
        # После simplify может быть уже NMS результат
        
        output = outputs[0]  # Основной выход
        
        # Если это результат после NMS (3 отдельных выхода: boxes, scores, classes)
        if len(outputs) == 3:
            boxes = outputs[0][0]  # [num_det, 4]
            scores = outputs[1][0]  # [num_det]
            classes = outputs[2][0].astype(int)  # [num_det]
        else:
            # Сырой формат [1, num_predictions, 85]
            output = output[0]  # Убираем batch dimension
            
            # Фильтруем по objectness/confidence
            if output.shape[1] == 6:  # [x, y, w, h, conf, class]
                conf_mask = output[:, 4] > score_threshold
                output = output[conf_mask]
                
                if len(output) == 0:
                    continue
                
                boxes_xywh = output[:, :4]
                scores = output[:, 4]
                classes = output[:, 5].astype(int)
            else:  # [x, y, w, h, obj, class0, class1, ...]
                objectness = output[:, 4]
                class_probs = output[:, 5:]
                
                # Находим максимальный класс
                class_scores = objectness[:, None] * class_probs
                max_scores = np.max(class_scores, axis=1)
                classes = np.argmax(class_scores, axis=1)
                
                # Фильтруем
                conf_mask = max_scores > score_threshold
                output = output[conf_mask]
                scores = max_scores[conf_mask]
                classes = classes[conf_mask]
                
                if len(output) == 0:
                    continue
                
                boxes_xywh = output[:, :4]
            
            # Декодируем xywh -> xyxy
            x_center = boxes_xywh[:, 0]
            y_center = boxes_xywh[:, 1]
            width = boxes_xywh[:, 2]
            height = boxes_xywh[:, 3]
            
            x1 = (x_center - width / 2) * w / input_size
            y1 = (y_center - height / 2) * h / input_size
            x2 = (x_center + width / 2) * w / input_size
            y2 = (y_center + height / 2) * h / input_size
            
            boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Scale boxes to original image size (если еще не масштабированы)
        if boxes.max() <= input_size:
            scale_x = w / input_size
            scale_y = h / input_size
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        # Process detections
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Apply padding
            cx1 = max(0, x1 - padding)
            cy1 = max(0, y1 - padding)
            cx2 = min(w, x2 + padding)
            cy2 = min(h, y2 + padding)
            
            if cx1 >= cx2 or cy1 >= cy2:
                continue
            
            # Extract crop
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
    result_file = os.path.join(crops_data_dir, "yolo_onnx_results.json")
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)
    
    print(f"Finished. {crop_index} crops saved")
    
    return {
        "status": "ok",
        "model_type": "yolo_onnx",
        "crops_saved": crop_index,
        "total_frames": len(frame_paths),
        "result_file": result_file
    }


import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from pathlib import Path
import glob
import os
import json
import time
from typing import List


def inference_ssd(
        raw_data_dir: str,
        crops_data_dir: str,
        model_weights: str,
        num_classes: int = 2,
        score_threshold: float = 0.5,
        input_size: int = 320,  # SSDLite320 по умолчанию использует 320
        padding: int = 30,
        device: str = "cpu",
        image_extensions: List[str] = None,
):
    """
    SSD inference function compatible with Detectron2 interface
    
    Args:
        raw_data_dir: Directory with input images
        crops_data_dir: Directory to save crops
        model_weights: Path to .pth model weights
        num_classes: Number of classes (including background)
        score_threshold: Confidence threshold for detections
        input_size: Input size for model (default 320 for SSDLite320)
        padding: Padding around detected boxes for crops
        device: "cpu" or "cuda"
        image_extensions: List of image extensions to process
    
    Returns:
        Dictionary with status and results
    """
    
    if image_extensions is None:
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    
    # ----------------------------
    # Setup device
    # ----------------------------
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # ----------------------------
    # Load model using the SAME function as in training
    # ----------------------------
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.models.detection.ssd import SSDClassificationHead
    
    print("Loading SSD model...")
    
    # Create model exactly as in get_model() from training code
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    
    if hasattr(model.head, 'classification_head'):
        existing_head = model.head.classification_head
        in_channels = []
        for layer in existing_head.module_list:
            if hasattr(layer[0], 'in_channels'):
                in_channels.append(layer[0].in_channels)
            else:
                in_channels.append(layer[0][0].in_channels)
    else:
        in_channels = [672, 480, 512, 256, 256, 64]

    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
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
    with torch.no_grad():
        for idx, fp in enumerate(frame_paths):
            if (idx + 1) % 10 == 0:
                print(f"Processing image {idx + 1}/{len(frame_paths)}")
            
            # Read image
            img_cv = cv2.imread(fp)
            if img_cv is None:
                print(f"Warning: Could not read image {fp}")
                continue
            
            h, w = img_cv.shape[:2]
            time_label = Path(fp).stem.split("_")[-1]
            
            # Convert to PIL and tensor
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_tensor = F.to_tensor(img_pil).to(device)
            
            # Run inference
            t0 = time.time()
            outputs = model([img_tensor])
            inference_ms = (time.time() - t0) * 1000
            
            # Parse outputs
            output = outputs[0]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            
            # Filter by score threshold
            mask = scores >= score_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Process each detection
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(int, box)
                
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Apply padding
                cx1 = max(0, x1 - padding)
                cy1 = max(0, y1 - padding)
                cx2 = min(w, x2 + padding)
                cy2 = min(h, y2 + padding)
                
                if cx1 >= cx2 or cy1 >= cy2:
                    continue
                
                # Extract crop
                crop = img_cv[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                
                # Save crop
                crop_filename = f"crop_{time_label}_{crop_index}.png"
                crop_path = os.path.join(crops_data_dir, crop_filename)
                
                if not cv2.imwrite(crop_path, crop):
                    continue
                
                # Store detection info
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
                    "class": int(label),
                    "speed": {"inference_ms": round(inference_ms, 2)}
                })
                
                crop_index += 1
    
    # ----------------------------
    # Save JSON
    # ----------------------------
    result_file = os.path.join(
        crops_data_dir,
        "ssd_results.json"
    )
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)
    
    print(f"Finished. {crop_index} crops saved")
    
    return {
        "status": "ok",
        "model_type": "ssd",
        "crops_saved": crop_index,
        "total_frames": len(frame_paths),
        "result_file": result_file
    }