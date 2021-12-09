# 3D CNN

# Getting Started
## Requeriments
* Python 3.7 or 3.8
* pytorchvideo
* torch 1.8 or above
* pytorch-lightning
* torchvision

You can find all installation procedure on pyvienv.intall

## Training a custom dataset
### Preparation
Put your video dataset inside data/video_data in this form --

```
+ data
        - annotation
        + video_data
                - folder_class1
                - folder_class2
                + folder_class3
                        - video1.mp4
                        - video2.mp4
                        - video3.mp4
                        - ...        
```

To generate train/val/test partitions from the dataset. Information will be saved on *annotation* folder.

```
python prepare_data.py
```

## Training

To train a model on a gpu:

```
python train.py
```

This program uses pretrained architectures from *Model Zoo* These are the models that has been tested:
* X3D architectures: x3d_m, x3d_s
* Resnet(2+1)D: r2plus1d_r50


