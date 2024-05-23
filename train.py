###########################
# Author: Lam Thai Nguyen #
###########################

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNET
from utils import save_checkpoint, load_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs


# Hyperparameters
LR = 1e-4
DEVICE = "cuda"
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_WORKERS = 4  # originally 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/val"
VAL_MASK_DIR = "data/val_masks"

train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)    

val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


def train(loader: DataLoader, model: UNET, optimizer, loss_fn, scaler):
    """
    Train one epoch
    """
    loop = tqdm(loader)
    
    for i, data in enumerate(loop):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
    
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loop.set_postfix(loss=loss.item())


def main():
    model = UNET(3, 1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, train_transform, val_transform, NUM_WORKERS, PIN_MEMORY)
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}")
        train(train_loader, model, optimizer, loss_fn, scaler)
        
        # save model
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)
        
        # check accuracy
        check_accuracy(val_loader, model, DEVICE)
        
        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model)


def debug():
    model = UNET(3, 1).to(DEVICE)
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    _, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, train_transform, val_transform, NUM_WORKERS, PIN_MEMORY)
    
    model.eval()
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            # visualize the predictions
            plt.subplot(1, 2, 1)
            plt.imshow(preds[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title("Prediction")
            plt.subplot(1, 2, 2)
            plt.imshow(y[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title("Ground Truth")
            plt.show()
            break
            
    model.train()
        
    
if __name__ == "__main__":
    debug()
    # main()
    