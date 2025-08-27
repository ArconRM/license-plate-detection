import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo


def train_detectron(
        train_dataset_name: str,
        val_dataset_name: str,
        config_yaml: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        output_dir: str = "./output",
        num_classes: int = 1,
        batch_size: int = 3,
        base_lr: float = 0.00025,
        max_iter: int = 20000,
        num_workers: int = 2,
        batch_size_per_image: int = 512,
        resume: bool = False,
):
    """
    Configures and trains a Faster R-CNN model using Detectron2.

    Args:
      train_dataset_name (str): Registered name of the training dataset.
      val_dataset_name (str): Registered name of the validation dataset.
      config_yaml (str): Detectron2 COCO config yaml file to use.
      output_dir (str): Directory to save outputs.
      num_classes (int): Number of classes to predict.
      batch_size (int): Images per batch in training.
      base_lr (float): Learning rate.
      max_iter (int): Maximum number of solver iterations.
      num_workers (int): Number of data loader workers.
      batch_size_per_image (int): Proposal batch size per image for ROI heads.
      resume (bool): Whether to resume from last checkpoint.

    Returns:
      None
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_yaml))
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_yaml)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.OUTPUT_DIR = output_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()
