import pydicom
import os
import pandas as pd
import numpy as np
import cv2
import time
import argparse
from ultralytics import YOLO
from config import TEST_DATA_PATH, BEST_MODEL_PATH, SUBMISSION_PATH


def create_submission(model: YOLO, dir_path: str) -> pd.DataFrame:
    """
    Create a submission DataFrame with patient IDs and corresponding prediction strings.

    Args:
        model (YOLO): YOLO model for object detection.
        dir_path (str): Path to the directory containing DICOM images.

    Returns:
        pd.DataFrame: DataFrame with 'patientId' and 'PredictionString' columns.
    """
    # Get a list of files in the specified directory and sort them
    file_list = os.listdir(dir_path)
    file_list = sorted(file_list)

    # Extract patient IDs from file names
    patient_ids = [file_name.replace(".dcm", "") for file_name in file_list]

    # List to store prediction strings
    results_list = []

    # Loop through each DICOM image in the directory
    for img in file_list:
        # Read the DICOM image
        image_path = os.path.join(dir_path, img)
        dicom_image = pydicom.dcmread(image_path)
        image = dicom_image.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Initialize an empty string to store the prediction string for the image
        img_string = ''

        # Get predictions from the YOLO model for the current image
        results = model(image)

        # Loop through each prediction in the results
        for i, result in enumerate(results):
            # Check if the prediction contains bounding box information
            if result.boxes.xyxy.numel() != 0:
                # Extract confidence and bounding box coordinates
                confidence = result.boxes.conf[i].item()
                confidence = np.around(confidence, 1)
                x_min, y_min, x_max, y_max = result.boxes.xyxy[i].squeeze(0)

                # Append bounding box information to the image string
                img_string += f"{confidence} {int(x_min)} {int(y_min)} {int(x_max - x_min)} {int(y_max - y_min)}"

        # Append the prediction string for the image to the results list
        results_list.append(img_string)

    # Create a DataFrame with patient IDs and prediction strings
    df = pd.DataFrame({'patientId': patient_ids, 'PredictionString': results_list})
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
