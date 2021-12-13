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

This program uses pretrained architectures from *Model Zoo*. These are the different models that can been used by calling the argument *--model*. If you want add more *Model Zoo* architectures please check the **Add a new Model** section:
* X3D architectures: x3d_m, x3d_s, x3d_xs
* Resnet(2+1)D: r2plus1d_r50

## Add a new model
If you are keen on use a new *Model Zoo*archicture that is not listed above, modify the **input_transformations_by_architecture.py** file. This file contains a dictionary in which the key is the name of the architecture and the values are transformation parameters.

You can find information about all available models here.


