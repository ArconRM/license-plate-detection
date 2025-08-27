from ultralytics import YOLO


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
