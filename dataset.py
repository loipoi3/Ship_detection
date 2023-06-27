import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2

class ShipDataset(Dataset):
    def __init__(self, train_csv, images_dir, image_size,  transform):
        super(ShipDataset, self).__init__()
        self.dataframe = train_csv
        self.images_dir = images_dir
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        # Get the filename and path of the image
        image_filename = self.dataframe.iloc[index]['ImageId']
        image_path = os.path.join(self.images_dir, image_filename)

        # Read and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)

        # Get the encoded masks for the image
        img_masks = self.dataframe.loc[self.dataframe['ImageId'] == image_filename, 'EncodedPixels'].tolist()

        # Combine the masks into a single mask
        all_masks = np.zeros(self.image_size)
        for mask in img_masks:
            all_masks += rle_decode(mask, self.image_size)

        if self.transform:
            if np.sum(all_masks)>0:
                # Apply data augmentation to the image and mask
                augmented = self.transform(image=image, mask=all_masks)
                image = augmented["image"]
                mask = augmented["mask"]
                m = torch.max(mask)
                return image.to(torch.float32), mask.to(torch.float32)

            # If the mask is empty, return the original image and mask
            return torch.from_numpy(np.transpose(image, (2,0,1))).to(torch.float32), torch.from_numpy(all_masks).to(torch.float32)

        # Return the original image and mask without augmentation
        return torch.from_numpy(np.transpose(image, (2,0,1))).to(torch.float32), torch.from_numpy(all_masks).to(torch.float32)

def rle_decode(mask_rle, shape):
    # Return the original image and mask without augmentation
    if isinstance(mask_rle, float) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1

    mask = mask.reshape((shape[1], shape[0]))
    mask = np.flipud(mask)
    mask = np.rot90(mask, k=-1)

    return mask