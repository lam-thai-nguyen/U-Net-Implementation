# Implementation of U-Net Convolutional Networks for Biomedical Image Segmentation

- Dataset Source: [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data).

- Tutorial YouTube Video: [PyTorch Image Segmentation Tutorial with U-NET: everything from scratch baby](https://www.youtube.com/watch?v=IHq1t7NxS8k&list=LL&index=1&t=2094s&ab_channel=AladdinPersson).

- After downloading the dataset, organize the data folder as follows:

```
.
└── data/
    ├── train/
    │   └── <5040 items>
    ├── train_masks/
    │   └── <5040 items>
    ├── val/
    │   └── <48 items>
    └── val_masks/
        └── <48 items>
```

- Result: 
  
```
Epoch 1
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 315/315 [1:23:08<00:00, 15.84s/it, loss=0.125]
=> Saving checkpoint
Accuracy: 99.12%
Dice score: 0.9806301593780518
```