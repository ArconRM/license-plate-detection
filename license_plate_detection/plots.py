import os
import cv2
import matplotlib.pyplot as plt

def draw_bboxes(image_path, annotations):
    """Draw bounding boxes on an image given COCO annotations."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = [int(coord) for coord in bbox]

        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        country = ann.get("attributes", {}).get("country", "Unknown")
        label = f"License Plate ({country})"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return image

def visualize_random_images(image_id_to_info, image_id_to_annotations, img_dir, n=5):
    """Show random images with bounding boxes."""
    import random
    random_ids = random.sample(list(image_id_to_info.keys()), min(n, len(image_id_to_info)))

    for img_id in random_ids:
        img_info = image_id_to_info[img_id]
        img_path = os.path.join(img_dir, img_info['file_name'])
        annotations = image_id_to_annotations.get(img_id, [])

        image_with_boxes = draw_bboxes(img_path, annotations)

        plt.figure(figsize=(10, 6))
        plt.imshow(image_with_boxes)
        plt.title(f"Image: {img_info['file_name']} (ID: {img_id})")
        plt.axis('off')
        plt.show()