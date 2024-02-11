import os
import re
import cv2
import pandas as pd
from pydicom import dcmread
from sklearn.model_selection import train_test_split


def save_data(folder: str, dataset: pd.DataFrame) -> None:
    """
    Save bounding box information in YOLO format and corresponding images.

    Args:
        folder (str): Path to the destination folder.
        dataset (DataFrame): Pandas DataFrame containing image information.

    Returns:
        None
    """
    # Extracting image paths from the DataFrame
    images_path = dataset['image_path'].tolist()

    # Iterating through each image path
    for img in images_path:
        # Extracting information for the current image
        image = dataset[dataset['image_path'] == img]

        # List to store YOLO-formatted bounding box information
        yolo_lines = []

        # Extracting bounding box information for the current image
        bboxes = image['bboxes'].tolist()
        for bb in bboxes:
            # Parsing bounding box information from the string
            b = bb.split(',')
            bboxes_list = [n.split('-') for n in b]
            bboxes_list.pop()
            for bbox in bboxes_list:
                # Converting bounding box information to YOLO format
                label = int(bbox[0])
                x_min = float(bbox[1])
                y_min = float(bbox[2])
                w = float(bbox[3])
                h = float(bbox[4])
                x_center = ((2 * x_min + w) / (2 * 1024))
                y_center = ((2 * y_min + h) / (2 * 1024))

                # Appending YOLO-formatted bounding box to the list
                yolo_lines.append([label, x_center, y_center, w / 1024, h / 1024])

        # Extracting patient ID from the image path
        match = re.search(r'/([\w-]+)\.dcm$', img)
        extracted_string = match.group(1)
        patient_id = extracted_string

        # Writing YOLO-formatted bounding box information to a text file
        with open(os.path.join(folder, "labels", patient_id + ".txt"), "w") as f:
            for bbox in yolo_lines:
                label, x_center, y_center, w, h = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
                f.write(f"{label} {x_center} {y_center} {w} {h}\n")

        # Reading DICOM image and converting to RGB format
        img_dcm = dcmread(img)
        img_rgb = cv2.cvtColor(img_dcm.pixel_array, cv2.COLOR_GRAY2RGB)

        # Saving the RGB image to the destination folder
        cv2.imwrite(os.path.join(folder, "images", f"{patient_id}.jpg"), img_rgb)


def create_data(dst_path: str, dataset: pd.DataFrame) -> None:
    """
    Creates training, validation, and test datasets in YOLO format.

    Args:
        dst_path (str): Path to the destination folder.
        dataset (DataFrame): Pandas DataFrame containing image information.

    Returns:
        None
    """
    train_folder = f"{dst_path}/train"
    val_folder = f"{dst_path}/val"
    test_folder = f"{dst_path}/test"
    # Split the annotation files into metrics, validation, and test sets
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, test_data = train_test_split(train_data, test_size=0.1, random_state=42)
    if not os.path.exists(train_folder):
        # Create metrics, validation, and test folders if they don't exist
        for folder in [train_folder, val_folder, test_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                os.makedirs(os.path.join(folder, "images"))
                os.makedirs(os.path.join(folder, "labels"))
        save_data(train_folder, train_data)
        save_data(val_folder, val_data)
        save_data(test_folder, test_data)
    else:
        print(f"Data created!")



