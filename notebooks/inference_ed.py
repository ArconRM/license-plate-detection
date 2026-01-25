import os
import glob
import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess


def inference_efficientdet(
    raw_data_dir: str,
    crops_data_dir: str,
    model_weights: str,
    class_names: List[str] = None,
    compound_coef: int = 1,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.2,
    input_size: int = None,
    padding: int = 30,
    device: str = "cuda",
    use_float16: bool = True,
    batch_size: int = 8,
    anchor_ratios: List[tuple] = None,
    anchor_scales: List[float] = None,
    image_extensions: List[str] = None,
):
    if image_extensions is None:
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    if class_names is None:
        class_names = ["license_plate"]

    if anchor_ratios is None:
        anchor_ratios = [(2.0, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0)]

    if anchor_scales is None:
        anchor_scales = [2 ** -1, 2 ** 0, 2 ** (1.0 / 3.0)]

    use_cuda = device == "cuda" and torch.cuda.is_available()

    if use_cuda:
        cudnn.benchmark = True
        cudnn.fastest = True

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    if input_size is None:
        input_size = input_sizes[compound_coef]

    num_classes = len(class_names)

    # ----------------------------
    # Load model
    # ----------------------------
    print(f"[INFO] Loading EfficientDet-D{compound_coef}")

    model = EfficientDetBackbone(
        compound_coef=compound_coef,
        num_classes=num_classes,
        ratios=anchor_ratios,
        scales=anchor_scales,
    )

    model.load_state_dict(torch.load(model_weights, map_location="cpu"))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()

    print(f"[INFO] Device: {device}, AMP: {use_float16}, Batch: {batch_size}")

    # ----------------------------
    # Collect images
    # ----------------------------
    frame_paths = []
    for ext in image_extensions:
        frame_paths.extend(
            glob.glob(os.path.join(raw_data_dir, "**", ext), recursive=True)
        )

    frame_paths = sorted(frame_paths)

    if not frame_paths:
        return {"status": "error", "message": "No images found"}

    os.makedirs(crops_data_dir, exist_ok=True)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    all_detections = []
    crop_index = 0

    # ----------------------------
    # Batch buffers
    # ----------------------------
    batch_imgs = []
    batch_metas = []
    batch_ori_imgs = []
    batch_paths = []

    cached_anchors = None

    def run_batch():
        nonlocal crop_index, cached_anchors

        x = torch.from_numpy(np.stack(batch_imgs)).to(device)
        x = x.permute(0, 3, 1, 2).contiguous()

        t0 = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_float16):
            features, regression, classification, anchors = model(x)

            # cache anchors ONCE
            if cached_anchors is None:
                cached_anchors = anchors

            out = postprocess(
                x,
                cached_anchors,
                regression,
                classification,
                regressBoxes,
                clipBoxes,
                score_threshold,
                iou_threshold,
            )

        infer_ms = (time.time() - t0) * 1000

        out = invert_affine(batch_metas, out)

        # ----------------------------
        # Process detections
        # ----------------------------
        for i, det in enumerate(out):
            ori_img = batch_ori_imgs[i]
            h, w = ori_img.shape[:2]
            time_label = Path(batch_paths[i]).stem.split("_")[-1]

            if len(det["rois"]) == 0:
                continue

            for j in range(len(det["rois"])):
                x1, y1, x2, y2 = det["rois"][j].astype(int)
                score = float(det["scores"][j])
                cls = int(det["class_ids"][j])

                if x1 >= x2 or y1 >= y2:
                    continue

                cx1 = max(0, x1 - padding)
                cy1 = max(0, y1 - padding)
                cx2 = min(w, x2 + padding)
                cy2 = min(h, y2 + padding)

                crop = ori_img[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                crop_filename = f"crop_{time_label}_{crop_index}.png"
                crop_path = os.path.join(crops_data_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

                all_detections.append({
                    "frame": batch_paths[i],
                    "crop_index": crop_index,
                    "crop_filename": crop_filename,
                    "coordinates": {
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                    },
                    "confidence": score,
                    "class": cls,
                    "class_name": class_names[cls],
                    "speed": {"inference_ms": round(infer_ms, 2)},
                })

                crop_index += 1

    # ----------------------------
    # Main loop
    # ----------------------------
    for idx, fp in enumerate(frame_paths):
        if (idx + 1) % 10 == 0:
            print(f"[INFO] Processing {idx + 1}/{len(frame_paths)}")

        ori_img = cv2.imread(fp)
        if ori_img is None:
            continue

        _, framed_imgs, framed_metas = preprocess(fp, max_size=input_size)

        batch_imgs.append(framed_imgs[0])
        batch_metas.append(framed_metas[0])
        batch_ori_imgs.append(ori_img)
        batch_paths.append(fp)

        if len(batch_imgs) == batch_size:
            run_batch()
            batch_imgs.clear()
            batch_metas.clear()
            batch_ori_imgs.clear()
            batch_paths.clear()

    # remaining images
    if batch_imgs:
        run_batch()

    # ----------------------------
    # Save JSON
    # ----------------------------
    result_file = os.path.join(
        crops_data_dir,
        f"efficientdet_d{compound_coef}_results.json",
    )

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved {crop_index} crops")

    return {
        "status": "ok",
        "model_type": f"efficientdet-d{compound_coef}",
        "crops_saved": crop_index,
        "total_frames": len(frame_paths),
        "result_file": result_file,
    }
