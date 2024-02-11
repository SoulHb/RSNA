import pydicom
import os
import pandas as pd
import numpy as np
import cv2
import time
import argparse
from ultralytics import YOLO
from config import TEST_DATA_PATH, BEST_MODEL_PATH, SUBMISSION_PATH

from ultralytics import YOLO

import os
import cv2
import pydicom
import numpy as np
import pandas as pd


def resize_bbox(bbox: tuple, scale_factor: float = 0.17) -> tuple:
    """
    Resize a bounding box based on a scale factor.

    Args:
        bbox (tuple): Tuple containing (x1, y1, x2, y2) coordinates of the bounding box.
        scale_factor (float): Scaling factor for resizing the bounding box.

    Returns:
        tuple: Resized bounding box coordinates (new_x1, new_y1, new_width, new_height).
    """
    # Extract coordinates from the bounding box
    x1, y1, x2, y2 = bbox
    # Calculate width and height of the bounding box
    width = x2 - x1
    height = y2 - y1

    # Calculate new width and height based on the scale factor
    new_width = width * (1 - scale_factor)
    new_height = height * (1 - scale_factor)

    # Calculate new x and y coordinates for the top-left corner of the resized bounding box
    new_x1 = x1 + (width - new_width) / 2
    new_y1 = y1 + (height - new_height) / 2

    # Return the coordinates of the resized bounding box
    return new_x1, new_y1, new_width, new_height


def create_submission(model: YOLO, dir_path: str) -> pd.DataFrame:
    """
    Create a submission DataFrame containing predictions for a given directory of RGB images.

    Args:
        model: Object representing the detection model used for predictions.
        dir_path (str): Path to the directory containing RGB images.

    Returns:
        pd.DataFrame: DataFrame with columns 'patientId' and 'PredictionString'.
                      'patientId' contains patient IDs, and 'PredictionString' contains
                      formatted prediction strings for bounding box coordinates and confidence.
    """
    # Get the list of files in the directory and sort them
    file_list = os.listdir(dir_path)
    file_list = sorted(file_list)
    # Extract patient IDs from file names
    patient_ids = [file_name.replace(".dcm", "") for file_name in file_list]
    results_list = []

    # Iterate through each image in the directory
    for img in file_list:
        # Construct the full path to the DICOM image
        image_path = os.path.join(dir_path, img)
        # Read the DICOM image
        dicom_image = pydicom.dcmread(image_path)
        # Extract pixel array from the DICOM image
        image = dicom_image.pixel_array
        # Convert grayscale image to BGR format
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Initialize an empty string to store predictions for the current image
        img_string = ''
        # Get predictions from the model for the current image
        results = model(image)

        # Iterate through each prediction in the results
        for i, result in enumerate(results):
            # Check if there are any bounding boxes in the current prediction
            if result.boxes.xyxy.numel() != 0:
                # Extract confidence score for the bounding box
                confidence = result.boxes.conf[i].item()
                confidence = np.around(confidence, 1)
                # Resize the bounding box coordinates using the resize_bbox function
                x_min, y_min, width, height = resize_bbox(result.boxes.xyxy[i].squeeze(0))
                # Append the formatted string to the img_string
                img_string += f"{confidence} {x_min} {y_min} {width} {height} "

        # Append the img_string for the current image to the results_list
        results_list.append(img_string.strip())

    # Create a DataFrame with patient IDs and corresponding prediction strings
    df = pd.DataFrame({'patientId': patient_ids, 'PredictionString': results_list})
    # Return the DataFrame
    return df


def main(args: dict) -> None:
    """
    Main function to generate a submission CSV file.

    Args:
        args (dict): Dictionary containing command-line arguments.
            - 'data_path' (str): Path for the data folder.
            - 'best_model_path' (str): Path for the best model checkpoint.
            - 'submission_path' (str): Path for the submission.csv file.

    Returns:
        None
    """
    # Initialize the YOLO model with the specified checkpoint
    model = YOLO(args['best_model_path'] if args['best_model_path'] else BEST_MODEL_PATH)
    # Record the start time
    start_time = time.time()

    # Create submission DataFrame using the specified data path or default TEST_DATA_PATH
    submission = create_submission(model, args['data_path'] if args['data_path'] else TEST_DATA_PATH)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Save the submission DataFrame to a CSV file using the specified path or default SUBMISSION_PATH
    submission.to_csv(
        args['submission_path'] if args['submission_path'] else os.path.join(SUBMISSION_PATH, 'submission.csv'),
        index=False)


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser()

    # Add command-line arguments for specifying paths
    parser.add_argument("--data_path", type=str, help='Specify path for data folder')
    parser.add_argument("--best_model_path", type=str, help='Specify path for best model')
    parser.add_argument("--submission_path", type=str, help='Specify path for submission.csv')

    # Parsing command-line arguments
    args = parser.parse_args()
    args = vars(args)

    # Calling the main function with parsed arguments
    main(args)
