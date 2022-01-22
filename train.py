import os
import logging
import argparse

import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import pytorch_lightning
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

import torchmetrics
from torchmetrics.functional import confusion_matrix

import pytorchvideo.data

from input_transformations_by_architecture import model_transform_params

from torchvision.transforms import Compose, Lambda

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

"""
This video classification example demonstrates how PyTorchVideo models, datasets and
transforms can be used with PyTorch Lightning module. Specifically it shows how a
simple pipeline to train a Resnet on the Kinetics video dataset can be built.
Don't worry if you don't have PyTorch Lightning experience. We'll provide an explanation
of how the PyTorch Lightning module works to accompany the example.
The code can be separated into three main components:
1. VideoClassificationLightningModule (pytorch_lightning.LightningModule), this defines:
    - how the model is constructed,
    - the inner train or validation loop (i.e. computing loss/metrics from a minibatch)
    - optimizer configuration
2. KineticsDataModule (pytorch_lightning.LightningDataModule), this defines:
    - how to fetch/prepare the dataset
    - the train and val dataloaders for the associated dataset
3. pytorch_lightning.Trainer, this is a concrete PyTorch Lightning class that provides
  the training pipeline configuration and a fit(<lightning_module>, <data_module>)
  function to start the training/validation loop.
All three components are combined in the train() function. We'll explain the rest of the
details inline.
"""

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        
        # Here we load the pretrained PyTorchVideo model.
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('facebookresearch/pytorchvideo', self.args.model, pretrained=True)  
        
        # finetuning modifications
        self.model.blocks[-1].proj = nn.Linear(2048, self.args.num_target_classes)
 
        # CUDA for PyTorch
        device = torch.device(f"cuda:{self.args.gpu}" if self.args.use_cuda else "cpu")
        
        self.model.to(device)
            
        # metrics
        self.train_loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy(num_classes=self.args.num_target_classes)
        self.fscore = torchmetrics.F1(num_classes=self.args.num_target_classes)
        self.recall = torchmetrics.Recall(num_classes=self.args.num_target_classes)
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=self.args.num_target_classes)
        
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

     
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = self.train_loss(y_hat, batch["label"])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        
        # Log the train loss to Tensorboard
        self.log("train_loss", loss)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
     
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("val_loss", loss)
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        self.starter.record()
        y_hat = self.model(batch["video"])
        self.ender.record()
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.test_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        rec = self.recall(F.softmax(y_hat, dim=-1), batch["label"])
        f1 = self.fscore(F.softmax(y_hat, dim=-1), batch["label"])
        cm = self.confmat(F.softmax(y_hat, dim=-1), batch["label"])
        
        inference_time = self.starter.elapsed_time(self.ender)
        itime = inference_time
        
        self.log("test_acc", acc)
        self.log("test_rec", rec)
        self.log("F1_score",f1)
        self.log("inference time", itime)
        
        print()
        print(cm)
        
        return loss

    def configure_optimizers(self):
        """
        We use the SGD optimizer with per step cosine annealing scheduler.
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.max_epochs, last_epoch=-1
        )
        return [optimizer], [scheduler]
    
    
class DataModule(pytorch_lightning.LightningDataModule):

    """
    This LightningDataModule implementation constructs a PyTorchVideo dataset for both
    the train and val partitions. It defines each partition's augmentation and
    preprocessing transforms and configures the PyTorch DataLoaders.
    """

    def __init__(self, args, model_transform_params):
        super().__init__()
        self.args = args
        self.transform_params = model_transform_params[args.model]
        
    def train_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train
        """
        train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                [
                    UniformTemporalSubsample(self.transform_params["num_frames"]),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(self.args.video_means, self.args.video_stds),
                    ShortSideScale(size=self.transform_params["side_size"]),
                    CenterCropVideo(
                        crop_size=(self.transform_params["crop_size"], self.transform_params["crop_size"])
                    )
                ]
               ),
             ),
          ]
        )
        
        if self.args.binary:
            filetype = "train_bi.csv"
        else:
            filetype = "train.csv"
			
        self.train_dataset = LimitDataset(
            pytorchvideo.data.labeled_video_dataset(
            data_path=os.path.join(self.args.annotation_path, filetype),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.args.clip_duration),
            decode_audio=False,
            transform = train_transform,
            )
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size= self.args.batch_size,
            num_workers= self.args.num_workers,
        )
       
    def val_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        val_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                [
                    UniformTemporalSubsample(self.transform_params["num_frames"]),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(self.args.video_means, self.args.video_stds),
                    ShortSideScale(size=self.transform_params["side_size"]),
                    CenterCropVideo(
                        crop_size=(self.transform_params["crop_size"], self.transform_params["crop_size"])
                    )
                ]
               ),
             ),
          ]
        )
 
        if self.args.binary:
            filetype = "val_bi.csv"
        else:
            filetype = "val.csv"
                    
        self.val_dataset = LimitDataset(
            pytorchvideo.data.labeled_video_dataset(
            data_path=os.path.join(self.args.annotation_path, filetype),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.args.clip_duration),
            decode_audio=False,
            transform = val_transform,
            )
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size= self.args.batch_size,
            num_workers= self.args.num_workers,
        ) 
        
    def test_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/test
        """
        test_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                [
                    UniformTemporalSubsample(self.transform_params["num_frames"]),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(self.args.video_means, self.args.video_stds),
                    ShortSideScale(size=self.transform_params["side_size"]),
                    CenterCropVideo(
                        crop_size=(self.transform_params["crop_size"], self.transform_params["crop_size"])
                    )
                ]
               ),
             ),
          ]
        )

        if self.args.binary:
            filetype = "test_bi.csv"
        else:
            filetype = "test.csv"
                    
        self.test_dataset = LimitDataset(
            pytorchvideo.data.labeled_video_dataset(
            data_path=os.path.join(self.args.annotation_path, filetype),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.args.clip_duration),
            decode_audio=False,
            transform = test_transform,
            )
        )
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size= self.args.batch_size,
            num_workers= self.args.num_workers,
        ) 
    
class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos
    

    
def main():
    """
    To train the Net with our dataset we construct the two modules above,
    and pass them to the fit function of a pytorch_lightning.Trainer.
    """
    setup_logger()

    pytorch_lightning.trainer.seed_everything(299)
    parser = argparse.ArgumentParser()
    
    # CUDA
    parser.add_argument("--use_cuda", action='store_true')
    parser.add_argument("--gpu", default = 0)
    
    # Model parameters.
    parser.add_argument("--model", default="x3d_m", type=str, help="Defines network's architecture. Tested models: x3d_xs, x3d_s, x3d_m, r2plus1d_r50")
    # parser.add_argument('--freeze', action='store_true', help='If true, freezes the pretrained network.') TBD
    parser.add_argument("--load_ckpt", action='store_true', help="If true, activates load-from-checkpoint mode. Requires add --ckpt_path and --hparams_path to work.")
    parser.add_argument("--ckpt_path", required=False, type=str, help="Checkpoint file path (.ckpt)")
    parser.add_argument("--hparams_path", required=False, type=str, help="Hyperparameters file path(.yaml)")  
        
    # Train parameters
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    
    # Data parameters.
    parser.add_argument("--video_path", default = './data/video_data/', type=str)
    parser.add_argument("--annotation_path", default = './data/annotation/', type=str)
    
    parser.add_argument("--clip_duration", default=2, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    
    
    parser.add_argument("--num_target_classes", default=5, type=int)
    parser.add_argument("--binary", action='store_true', help="If true, loads binary data ") # binary data

    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    
    
    # Test mode
    parser.add_argument("--test", action='store_true', help="If true, activates test mode. Requires add --ckpt_path and --hparams_path to work.")
    
    # Trainer parameters.
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=100,
        callbacks=[LearningRateMonitor()],
        replace_sampler_ddp=False,
        reload_dataloaders_every_epoch=False,
    )
    
    # Build trainer, ResNet lightning-module and data-module.
    args = parser.parse_args()
    
    if args.test:
        test_mode(args)
    else:
        train_mode(args)
    
#------------------------------------ train --------------------------- 
def train_mode(args):
    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    filename= args.model + "-" + "{epoch}-{val_loss:.3f}"
    )
    trainer = pytorch_lightning.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    data_module = DataModule(args, model_transform_params)
    if args.load_ckpt:
        classification_module = VideoClassificationLightningModule(args)
        trainer.fit(classification_module, data_module, ckpt_path = args.ckpt_path)
    else:	
        classification_module = VideoClassificationLightningModule(args)
        trainer.fit(classification_module, data_module)
    
def test_mode(args):
    model = VideoClassificationLightningModule.load_from_checkpoint(args.ckpt_path, hparams_file=args.hparams_path)
    trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    data_module = DataModule(args, model_transform_params)
    trainer.test(model, dataloaders = data_module)
    
#------------------------------------------------------------------------

def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
        
if __name__ == "__main__":
    main()
