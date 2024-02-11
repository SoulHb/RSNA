import pandas as pd
import os
import argparse
from config import *
from dataset import create_data
from ultralytics import YOLO


def main(args: dict) -> None:
    """
    Main function to process and train YOLO model on medical image data.

    Args:
        args (dict): Dictionary containing command-line arguments.
            - 'data_path' (str): Path for the data folder.
            - 'yaml_path' (str): Path for dataset.yaml.
            - 'epochs' (int): Number of epochs for model training.
            - 'batch_size' (int): Batch size for training.

    Returns:
        None
    """
    # Load detailed class information and train labels from CSV files
    class_data = pd.read_csv(
        os.path.join(args['data_path'], 'stage_2_detailed_class_info.csv') if args['data_path'] else CLASS_DATA_PATH)
    train_labels = pd.read_csv(
        os.path.join(args['data_path'], 'stage_2_train_labels.csv') if args['data_path'] else TRAIN_LABELS_PATH)

    # Combine class data and train labels into a single dataset
    dataset = pd.concat([class_data.drop(columns='patientId'), train_labels], axis=1)

    # Filter the dataset to include only 'Lung Opacity' class
    dataset = dataset[dataset["class"] == "Lung Opacity"]

    # Construct the full path to the image files in the dataset
    image_directory = args['data_path'] if args['data_path'] else DATA_FOLDER
    dataset['image_path'] = f'{image_directory}/stage_2_train_images/' + dataset['patientId'] + '.dcm'

    # Use specified YAML path or use the default path from the configuration
    data_yaml = args['yaml_path'] if args['yaml_path'] else YAML_PATH

    # Use specified number of epochs or use the default value from the configuration
    epochs = args['yaml_path'] if args['yaml_path'] else EPOCHS

    # Use specified batch size or use the default value from the configuration
    batch_size = args['batch_size'] if args['batch_size'] else BATCH_SIZE

    # Mapping of class labels
    class_mapping = {'Lung Opacity': 0}

    # Create a list to store image paths and corresponding bounding boxes
    df = []
    data_list = dataset["image_path"].values.tolist()
    past = []

    # Loop through each image path in the dataset
    for img_path in data_list:
        if img_path in past:
            continue
        images = dataset[dataset['image_path'] == img_path]
        bboxes = ''

        # Iterate over each row (bounding box) in the images DataFrame
        for _, row in images.iterrows():
            label = class_mapping[row['class']]
            x_min = row['x']
            y_min = row['y']
            width = row['width']
            height = row['height']
            bboxes += f'{label}-{x_min}-{y_min}-{width}-{height},'

        past.append(img_path)
        df.append({'image_path': img_path, 'bboxes': bboxes})

    # Create a new DataFrame with image paths and bounding boxes
    dataset = pd.DataFrame(df)

    # Use the create_data function to organize the dataset
    create_data(image_directory, dataset)

    # Initialize YOLO model using the specified checkpoint file
    model = YOLO('best.pt')

    # Train the YOLO model with specified parameters
    model.train(data=data_yaml, epochs=epochs, imgsz=1024, batch=batch_size)


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser()

    # Add command-line arguments for specifying paths and parameters
    parser.add_argument("--data_path", type=str, help='Specify path for data folder')
    parser.add_argument("--yaml_path", type=str, help='Specify path for dataset.yaml')
    parser.add_argument("--epochs", type=int, help='Specify epoch for model training')
    parser.add_argument("--batch_size", type=int, help='Specify batch size for training')

    # Parsing command-line arguments
    args = parser.parse_args()
    args = vars(args)

    # Calling the main function with parsed arguments
    main(args)
