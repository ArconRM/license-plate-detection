from ultralytics import YOLO

def train_yolo_model(model_path, data_path, epochs=20, img_size=768):
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
        task='detect',
        mode='train',
        data=data_path,
        epochs=epochs,
        imgsz=img_size
    )
