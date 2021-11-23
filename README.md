# 3D CNN

# Getting Started
## Requeriments
* Python3.9
* pytorch
* pytorchvideo

You can find all installation procedure on pyvienv.intall

## Training a custom dataset
### Preparation
Put your video dataset inside data/video_data in this form --
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

To generate train/val/test partitions from the dataset. Information will be saved on *annotation* folder. 
        python prepare_data.py
