## RSNA Pneumonia Detection Challenge

### Overview
This documentation provides information about the method of solving RSNA Pneumonia Detection Challenge kaggle competition, including the data used, the methods and ideas employed, and the metrics achieved. It also includes usage instructions and author information.


### Data
In this challenge competitors are predicting whether pneumonia exists in a given image. They do so by predicting bounding boxes around areas of the lung. Samples without bounding boxes are negative and contain no definitive evidence of pneumonia. Samples with bounding boxes indicate evidence of pneumonia.
The dataset used for training and scoring is loaded with pytorch.


[Link to the dataset on Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview)
## Model Architecture
The RSNA Pneumonia Detection Challenge neural network model is built using the YOLOv8n architecture.

## Method
Train YOLOv8n on 100 epochs and scale bboxes to 0.17 and create submission.csv

## Score
After training, the best model achieved 0.11939 on private score and 0.04166 public score.
## Usage
### Requirements
- Ubuntu 20.04
- Python 3.10

### Getting Started
Clone repository
```bash
git clone https://github.com/SoulHb/RSNA.git
```
Move to project folder
```bash
cd RSNA
```
Create conda env 
```bash
conda create --name=rsna python=3.10
```
Activate virtual environment
```bash
conda activate rsna 
```
Install pip 
```bash
conda install pip 
```
Install dependencies
```bash
pip install -r requirements.txt
```
### Training
The model is trained on the provided dataset using the following configuration:
- Optimizer: AdamW
- Learning rate: 0.002
- Batch size: 16
- Number of epochs: 100

Move to src folder
```bash
cd src
```
Run train.py
```bash
python train.py --data_path /path/to/data/folder --yaml_path /path/to/dataset.yaml --epochs 100 --batch_size 16

```

## Inference
To use the trained model for RSNA Pneumonia Detection Challenge, follow the instructions below:

Move to src folder
```bash
cd src
```
Run inference
```bash
python inference.py --data_path /path/to/data/folder --best_model_path /path/to/best/model --submission_path /path/to/submission.csv
```

## Author
This RSNA Pneumonia Detection Challenge project was developed by Namchuk Maksym. If you have any questions, please contact me: namchuk.maksym@gmail.com
