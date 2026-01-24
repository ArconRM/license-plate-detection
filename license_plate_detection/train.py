from click.core import batch
from ultralytics import YOLO

import os
# from detectron2.engine import DefaultTrainer
# from detectron2.config import get_cfg
# from detectron2 import model_zoo


def train_yolo_model(model_path, data_path, device, epochs=20, img_size=640, batch=1):
    """
    Train a YOLOv11 detection model with given parameters.

    Args:
      model_path (str): Path or name of the pretrained YOLOv11 model weights file.
      data_path (str): Path to the data YAML configuration file.
      epochs (int): Number of training epochs.
      img_size (int): Image size for training.
    """
    model = YOLO(model_path)
    model.train(
        device=device,
        task='detect',
        mode='train',
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        amp=True,
        verbose=False,
        workers=0 
    )


# def train_detectron(
#         train_dataset_name: str,
#         val_dataset_name: str,
#         config_yaml: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
#         output_dir: str = "./output",
#         num_classes: int = 1,
#         batch_size: int = 3,
#         base_lr: float = 0.0025,
#         max_iter: int = 20000,
#         num_workers: int = 2,
#         batch_size_per_image: int = 512,
#         use_amp = True,
#         resume: bool = False,
# ):
#     """
#     Configures and trains a Faster R-CNN model using Detectron2.

#     Args:
#       train_dataset_name (str): Registered name of the training dataset.
#       val_dataset_name (str): Registered name of the validation dataset.
#       config_yaml (str): Detectron2 COCO config yaml file to use.
#       output_dir (str): Directory to save outputs.
#       num_classes (int): Number of classes to predict.
#       batch_size (int): Images per batch in training.
#       base_lr (float): Learning rate.
#       max_iter (int): Maximum number of solver iterations.
#       num_workers (int): Number of data loader workers.
#       batch_size_per_image (int): Proposal batch size per image for ROI heads.
#       use_amp: Use mixed precision
#       resume (bool): Whether to resume from last checkpoint.

#     Returns:
#       None
#     """
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(config_yaml))

#     # --- ДАННЫЕ ---
#     cfg.DATASETS.TRAIN = (train_dataset_name,)
#     cfg.DATASETS.TEST = (val_dataset_name,)
#     cfg.DATALOADER.NUM_WORKERS = num_workers
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_yaml)

#     # --- ЖЕЛЕЗО: 3050 4GB VRAM ---
#     # 4GB - это очень мало. Батч строго 2.
#     cfg.SOLVER.IMS_PER_BATCH = batch_size
#     # Расчет LR: 0.02 для батча 16 -> 0.0025 для батча 2
#     cfg.SOLVER.BASE_LR = base_lr * batch_size / 16  # ✅
#     cfg.SOLVER.WARMUP_ITERS = 500

#     # Экономия памяти видеокарты
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image  # Стандарт 512, уменьшаем для VRAM
#     cfg.MODEL.BACKBONE.FREEZE_AT = 2  # Замораживаем первые слои (стандарт, но важно не менять)
#     cfg.SOLVER.AMP.ENABLED = use_amp  # Mixed Precision обязателен для 4GB

#     # --- СКОРОСТЬ CPU ИНФЕРЕНСА (КРИТИЧНО) ---
#     # Уменьшаем разрешение входа. 800px на CPU будет <1 FPS.
#     # Ставим динамический трейн и фикс тест 640px.
#     cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640)
#     cfg.INPUT.MAX_SIZE_TRAIN = 1000
#     cfg.INPUT.MIN_SIZE_TEST = 640
#     cfg.INPUT.MAX_SIZE_TEST = 1000

#     # Оптимизация RPN (Region Proposal Network)
#     # Это главное бутылочное горлышко на CPU.
#     # Снижаем кол-во регионов до NMS и после.
#     # Для большинства задач 500 финальных пропозалов за глаза (дефолт 1000).
#     cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000  # Дефолт 6000
#     cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500  # Дефолт 1000 - сильно ускорит вторую стадию

#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
#     cfg.SOLVER.MAX_ITER = max_iter
#     cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))
#     cfg.OUTPUT_DIR = output_dir

#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#     trainer = DefaultTrainer(cfg)
#     trainer.resume_or_load(resume=resume)
#     trainer.train()


# def train_detectron_retinanet(
#         train_dataset_name: str,
#         val_dataset_name: str,
#         config_yaml: str = "COCO-Detection/retinanet_R_50_FPN_1x.yaml",
#         output_dir: str = "./output",
#         num_classes: int = 1,
#         batch_size: int = 4,
#         base_lr: float = 0.0025,
#         max_iter: int = 20000,
#         num_workers: int = 2,
#         use_amp: bool = True,
#         resume: bool = False,
# ):
#     """
#     Configures and trains a RetinaNet model using Detectron2.

#     Args:
#       train_dataset_name (str): Registered name of the training dataset.
#       val_dataset_name (str): Registered name of the validation dataset.
#       config_yaml (str): Detectron2 COCO config yaml file to use.
#       output_dir (str): Directory to save outputs.
#       num_classes (int): Number of classes to predict.
#       batch_size (int): Images per batch in training.
#       base_lr (float): Learning rate.
#       max_iter (int): Maximum number of solver iterations.
#       num_workers (int): Number of data loader workers.
#       use_amp: Use mixed precision.
#       resume (bool): Whether to resume from last checkpoint.

#     Returns:
#       None
#     """
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(config_yaml))

#     cfg.DATASETS.TRAIN = (train_dataset_name,)
#     cfg.DATASETS.TEST = (val_dataset_name,)
#     cfg.DATALOADER.NUM_WORKERS = num_workers
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_yaml)

#     # --- ЖЕЛЕЗО: 3050 4GB VRAM ---
#     # RetinaNet R50 жрет память как не в себя из-за больших feature maps.
#     cfg.SOLVER.IMS_PER_BATCH = batch_size
#     cfg.SOLVER.BASE_LR  = base_lr
#     cfg.SOLVER.AMP.ENABLED = use_amp

#     # --- СКОРОСТЬ CPU ИНФЕРЕНСА ---
#     # RetinaNet очень чувствительна к размеру. 640px - разумный предел для CPU.
#     cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 576, 608, 640)
#     cfg.INPUT.MAX_SIZE_TRAIN = 1000
#     cfg.INPUT.MIN_SIZE_TEST = 640
#     cfg.INPUT.MAX_SIZE_TEST = 1000

#     # --- OPTIMIZATION ZOO ---
#     cfg.MODEL.RETINANET.NUM_CLASSES = num_classes

#     # 1. Срезаем лишние якоря (Anchors) если объекты не экстремально вытянутые.
#     # 3 ratio * 3 scale = 9 якорей на пиксель. Это много вычислений.
#     # Оставляем стандарт, но имей в виду: если у тебя нет очень узких объектов, убери 0.5 и 2.0.
#     # cfg.MODEL.RETINANET.ASPECT_RATIOS = [[1.0]] # Экстремальная оптимизация (если подходит под данные)

#     # 2. Фильтрация вывода (КРИТИЧНО ДЛЯ СКОРОСТИ)
#     # Порог уверенности 0.05 (твой конфиг) заставит CPU обрабатывать тысячи рамок.
#     # Поднимаем до 0.25. Модель не будет тратить время на "мусор".
#     cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.25

#     # 3. TopK Candidates
#     # Дефолт 1000. Снижаем до 300.
#     # Мы берем только 300 лучших боксов для NMS. Это линейно ускоряет пост-процессинг.
#     cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 300

#     cfg.SOLVER.MAX_ITER = max_iter
#     cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))
#     cfg.OUTPUT_DIR = output_dir

#     cfg.TEST.EVAL_PERIOD = 1000

#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#     trainer = DefaultTrainer(cfg)
#     trainer.resume_or_load(resume=resume)
#     trainer.train()
