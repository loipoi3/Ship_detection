import argparse
import torch
import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import unet
from albumentations.pytorch import ToTensorV2

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_path", type=str, help='Specify path for saving model')
    parser.add_argument("--image_path", type=str, help='Specify image for testing')
    args = parser.parse_args()
    args = vars(args)

    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read and preprocess the input image
    model = unet()
    model.load_state_dict(torch.load(args['saved_model_path'] if args['saved_model_path'] else "./models/model_epoch_1.pth"))
    model.to(device)
    model.eval()

    # Define the image transformation pipeline using Albumentations
    transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), ToTensorV2()])

    # Read and preprocess the input image
    image = cv2.imread(args["image_path"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (768, 768))
    Input = transform(image=image)["image"].unsqueeze(0).to(device=torch.device('cuda'), dtype=torch.float32)

    # Perform inference on the input image
    with torch.no_grad():
        prediction = model(Input.to(device))
    prediction = prediction.cpu().detach().numpy()
    prediction = np.transpose(prediction, (0, 2, 3, 1))
    prediction = np.squeeze(prediction, axis=0)

    # Visualize the original image and the predicted mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[1].imshow(prediction, cmap='gray')
    axes[1].set_title('Mask')
    plt.tight_layout()
    plt.show()