###########################
# Author: Lam Thai Nguyen #
###########################

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image, mask = augmentations["image"], augmentations["mask"]
            
        return image, mask
    
    
def test():
    print("Testing...")
    dataset = CarvanaDataset("data/train", "data/train_masks")
    assert len(dataset) == 5040
    assert dataset[0] is not None
    print("Test passed.")
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(dataset[0][0])
    plt.title("Raw image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(dataset[0][1], cmap='gray')
    plt.title("Mask image")
    plt.axis('off')
    
    plt.show()
    

if __name__ == "__main__":
    test()
    