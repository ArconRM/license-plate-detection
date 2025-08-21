import os
import json
import random
import shutil
import gdown
import zipfile
from tqdm import tqdm

from loguru import logger

def download_dataset(url, output_path="data/Detection.zip"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdown.download(url, output_path, quiet=False)


def unzip_dataset(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        root = os.path.commonprefix(names).rstrip('/')
        for member in zf.infolist():
            original = member.filename
            if original.startswith(root + "/"):
                member.filename = original[len(root) + 1:]
                if member.filename:
                    zf.extract(member, extract_to)

    logger.info(f"Extracted {zip_path} to {extract_to}")


def process_coco_dataset(in_base, out_base, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Process COCO dataset: load annotations, split into train/val/test, 
    filter annotations, and copy images to respective directories.
    
    Args:
        in_base (str): Base input directory
        out_base (str): Base output directory for splits
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing split information
    """
    # Load COCO annotations
    with open(os.path.join(in_base, "annotations/instances_default.json"), 'r') as f:
        coco_data = json.load(f)
    
    # Split dataset
    random.seed(seed)
    images = coco_data["images"]
    random.shuffle(images)

    n = len(images)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }
    
    os.makedirs(out_base, exist_ok=True)
    
    for split, imgs in splits.items():
        # Filter annotations for this split
        img_ids = {img["id"] for img in imgs}
        anns = [ann for ann in coco_data["annotations"] if ann["image_id"] in img_ids]
        
        split_json = {
            "images": imgs,
            "annotations": anns,
            "categories": coco_data["categories"],
        }
        
        # Save COCO json
        out_json_path = os.path.join(out_base, f"{split}.json")
        with open(out_json_path, "w") as f:
            json.dump(split_json, f)

        # Copy images
        img_dir = os.path.join(in_base, "images", "default")
        split_img_dir = os.path.join(out_base, "images", split)
        os.makedirs(split_img_dir, exist_ok=True)

        for img in tqdm(imgs, desc=f"Copying {split}"):
            src = os.path.join(img_dir, img["file_name"])
            dst = os.path.join(split_img_dir, img["file_name"])
            if os.path.exists(src):
                shutil.copy(src, dst)

        logger.info(f"{split}: {len(imgs)} images, {len(split_json['annotations'])} annotations")
    
    return splits