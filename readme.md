# Airbus Ship Detection Documentation

## Overview
This documentation provides information about the Airbus Ship Detection project, including the data used, the methods and ideas employed, and the accuracy achieved. It also includes usage instructions and author information.

## Data
The dataset used for training and scoring is loaded with pytorch: https://www.kaggle.com/competitions/airbus-ship-detection.

## Model Architecture
The Airbus Ship Detection neural network model is built using the UNET architecture. The architecture of the model consists two parts: encoder and decoder.

## Training
The model is trained on the provided dataset using the following configuration:
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: BCEWithLogitsLoss
- Batch size: 4
- Number of epochs: 

During training, accuracy and loss are tracked to track the performance of the model.

## Accuracy
After training, the model achieved an accuracy of '''91%''' on the validation set. This accuracy represents the model's ability to correctly classify VIN characters based on the provided dataset.

## Usage
To use the trained model for VIN classification, follow the instructions below:

1. First go to the project folder using cmd.
2. Next install virtualenv, write the following command and press Enter:
```bash
pip install virtualenv
```
3. Next create a new environment, write the following command and press Enter:
```bash
virtual_env name_of_the_new_env
```
### Example:
```bash
virtual_env ship
```
4. Next activate the new environment, write the following command and press Enter:
```bash
name_of_the_new_env\Scripts\activate
```
5. Write the following command and press Enter:
 ```bash
pip install -r requirements.txt
```
6. After installing all the libraries, type the following command and press Enter:
 ```bash
python inference.py --saved_model_path model_path --image_path image_path
```
### Example:
```bash
python inference.py --saved_model_path D:/projects/Winstars/model_epoch_0.pth --image_path D:/projects/Winstars/test/00a1aab5b.jpg
```

## Author
This Airbus Ship Detection project was developed by Dmytro Khar. If you have any questions or need further assistance, please contact qwedsazxc8250@gmail.com.