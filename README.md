# U-Net CNN for Biomedical Image Segmentation (From Scratch)

- [paper](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) | [dataset](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data) | [tutorital](https://www.youtube.com/watch?v=IHq1t7NxS8k&list=LL&index=1&t=2094s&ab_channel=AladdinPersson)

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

- It took too long so I only run for 1 epoch.