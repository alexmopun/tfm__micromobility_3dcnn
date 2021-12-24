# 3D CNN

# Getting Started
## Requeriments
To be able to train and test models:
* Python 3.7 or above
* pytorchvideo
* torch 1.8 or above
* pytorch-lightning
* torchvision

Additionally, if you want to make inference over a video:
* opencv

You can find all installation procedure on pyvienv.install

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

To generate train/val/test partitions from the dataset.

```
python prepare_data.py
```
Information will be saved on *annotation* folder.

## Training
This program uses pretrained architectures from *Model Zoo*. These are the different models that can been used by calling the argument **--model**. If you want add more *Model Zoo*'s architectures please check the **Add a new Model** section. These are the architectures that have been tested:
* X3D architectures: x3d_m, x3d_s, x3d_xs
* Resnet(2+1)D: r2plus1d_r50

### Training from scratch
To train a model on a gpu example:

```
srun --mem 8G gres=gpu:1 -c 4 --time=23:55:00 python train.py --model=x3d_s --use_cuda --gpus=1 --num_workers=4 --max_epochs=150
```
*c* must be equal to *num_workers*. If you are using *calcula* servers, please use *c* <= 4

### Continue training from checkpoint
The framework saves the checkpoint with minimum validation loss automatically. By default it will be savd inside the *lightning_logs* folder (it's the .ckpt file). Additionally it creates a *.yaml* file which contains the hyper-parameters. To continue the training from a checkpoint:

```
python train.py --model=x3d_s --use_cuda --gpus=1 --num_workers=4 --max_epochs=150 --load_ckpt --ckpt_path=path/to/checkpoint --hparams_path=path/to/hpamarams
```

### Test mode
To make inference of the test dataset:
```
python train.py --model=x3d_s --use_cuda --gpus=1 --num_workers=4 --test --ckpt_path=path/to/checkpoint --hparams_path=path/to/hpamarams
```

## Add a new model
If you are keen on use other *Model Zoo* archictures that is not listed above, modify the **input_transformations_by_architecture.py** file. This file contains a dictionary in which the key is the name of the architecture and the values are transformation parameters. It is possible that additional code modifications may be required to make the new model work.

You can find information about all available models [here](https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html).
