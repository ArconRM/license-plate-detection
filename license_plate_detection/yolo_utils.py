import os, shutil
from ultralytics.data.converter import convert_coco

from loguru import logger

def convert_coco_to_yolo(coco_ann_dir, yolo_base):
    print(f"Converting COCO dataset in {coco_ann_dir} to YOLO format at {yolo_base}")

    convert_coco(
        labels_dir=coco_ann_dir,
        save_dir=yolo_base,
        use_segments=False,
        use_keypoints=False,
        cls91to80=False,
    )

    # Copy images into YOLO directory
    src_img_dir = os.path.join(coco_ann_dir, "images")
    dst_img_dir = os.path.join(yolo_base, "images")

    for split in ["train", "val", "test"]:
        src_split_dir = os.path.join(src_img_dir, split)
        dst_split_dir = os.path.join(dst_img_dir, split)
        os.makedirs(dst_split_dir, exist_ok=True)

        for fname in os.listdir(src_split_dir):
            shutil.copy(os.path.join(src_split_dir, fname),
                        os.path.join(dst_split_dir, fname))

    logger.info("YOLO dataset ready at:", yolo_base)
