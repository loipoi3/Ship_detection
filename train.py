import os
import pandas as pd
import argparse
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from torch import nn
from albumentations.pytorch import ToTensorV2
from dataset import ShipDataset
from sklearn.model_selection import train_test_split
from model import unet
import numpy as np
import torchvision
import torchvision.transforms.functional as F

# Training loop
def train_loop(model, loss, optimizer, train_loader, device):
    model.train()
    for i, data in enumerate(train_loader):
        X = data[0].to(device) # Move input to device (e.g., GPU)
        y = data[1].to(device) # Move target to device
        y = y.unsqueeze(1) # Add a channel dimension to target
        X.requires_grad_(True) # Set requires_grad to True for input
        y.requires_grad_(True) # Set requires_grad to True for target
        predicted = model(X) # Forward pass
        Loss = loss(predicted, y) # Calculate the loss
        optimizer.zero_grad() # Clear gradients
        Loss.backward() # Clear gradients
        optimizer.step() # Clear gradients

        # Calculate true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(predicted > 0.5, y.to(torch.int64), mode='binary', num_classes=1)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro") # Calculate F1 score
        print(f"train_loss: {Loss.item()}, f1_score: {f1_score}")

# Validation loop
def val_loop(model, val_loader, loss, device):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X = data[0].to(device)
            y = data[1].to(device)
            y = y.unsqueeze(1)
            X.requires_grad_(True)
            y.requires_grad_(True)
            predicted = model(X)
            Loss = loss(predicted, y)
    tp, fp, fn, tn = smp.metrics.get_stats(predicted > 0.5, y.to(torch.int64), mode='binary', num_classes=1)
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    print(f"train_loss: {Loss.item()}, f1_score: {f1_score}")

def main(args):

    # Read arguments or use default values
    saved_model_path = args['saved_model_path'] if args['saved_model_path'] else "."
    train_root = args["train_root"] if args["train_root"] else r'./airbus-ship-detection/train_v2'
    train_csv = args["train_csv"] if args["train_csv"] else r'./airbus-ship-detection/train_ship_segmentations_v2.csv'
    epochs = args["epochs"] if args["epochs"] else 1000
    lr = args["lr"] if args["lr"] else 0.001
    image_size = (768, 768)

    if not os.path.exists(saved_model_path + "/models"):
        os.makedirs(saved_model_path + "/models")

    # Load and preprocess the dataset
    masks = pd.read_csv(train_csv)
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
    masks.drop(['ships'], axis=1, inplace=True)
    samples_per_group = 5000
    balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(samples_per_group) if len(x) > samples_per_group else x)
    train_ids, valid_ids = train_test_split(balanced_train_df, test_size=0.2, stratify=balanced_train_df['ships'])
    train_ids, test_ids = train_test_split(train_ids, test_size=0.2, stratify=train_ids['ships'])
    train_df = pd.merge(masks, train_ids)
    test_df = pd.merge(masks, test_ids)
    train_images_dir = train_root

    # Define data transformations
    transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), ToTensorV2()])

    # Create dataset instances
    train_dataset = ShipDataset(train_df, train_images_dir, image_size=image_size, transform=transform)
    test_dataset = ShipDataset(test_df, train_images_dir, image_size=image_size, transform=transform)

    # Create data loaders
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the UNet model
    model = unet()
    model.to(device)

    # Define loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    # Training loop
    for i in range(epochs):
        print(f'Epoch: {i}')
        train_loop(model, loss_fn, optimizer, train_loader, device)
        val_loop(model, test_loader, loss_fn, device)
        torch.save(model.state_dict(), f'{saved_model_path}/models/model_epoch_{i}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, help='Specify path for train dataset')
    parser.add_argument("--test_root", type=str, help='Specify path for test dataset')
    parser.add_argument("--train_csv", type=str, help='Specify path for csv file')
    parser.add_argument("--saved_model_path", type=str, help='Specify path for saving model')
    parser.add_argument("--batch_size", type=int, help='Specify batch size')
    parser.add_argument("--epochs", type=int, help='Specify epochs')
    parser.add_argument("--lr", type=float, help='Specify learning rate')
    args = parser.parse_args()
    args = vars(args)
    main(args)