import os
import numpy as np
from PIL import Image

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    train_normal_dir = os.path.join(data_dir, "train", "NORMAL")
    train_pneumonia_dir = os.path.join(data_dir, "train", "PNEUMONIA")

    # Create directories
    os.makedirs(train_normal_dir, exist_ok=True)
    os.makedirs(train_pneumonia_dir, exist_ok=True)

    print("Generating synthetic images...")

    # Generate NORMAL images (lower brightness, e.g. normal chest cavity)
    for i in range(40):
        # Create an image with a dark circle or dark background
        img_arr = np.random.randint(40, 100, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        img.save(os.path.join(train_normal_dir, f"normal_{i}.jpg"))

    # Generate PNEUMONIA images (higher brightness, e.g. white fluid/infiltrates in lung cavity)
    for i in range(40):
        img_arr = np.random.randint(140, 220, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        img.save(os.path.join(train_pneumonia_dir, f"pneumonia_{i}.jpg"))

    print(f"Generated 40 NORMAL images in: {train_normal_dir}")
    print(f"Generated 40 PNEUMONIA images in: {train_pneumonia_dir}")

if __name__ == "__main__":
    main()
