# IMPORTS
import argparse
import torch
import cv2
import numpy as np
import torch.nn.functional as F

from train import VideoClassificationLightningModule

from pytorchvideo.data.encoded_video import EncodedVideo

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
	
def extract_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
	
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = frame_count/fps
    
    cap.release()
	
    return duration, frame_count, fps
	
	
def inference(model, video_path, label, model_transform_params):
	
    model = model.cuda()

    # for inference time test
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
	
    # extracts video information
    duration, frame_count, fps = extract_video_info(video_path)
	
    #define transformation parameters
    transform_params = model_transform_params
    video_means = (0.45, 0.45, 0.45)
    video_stds = (0.225, 0.225, 0.225)

    transform = Compose(
        [
        ApplyTransformToKey(
          key="video",
          transform=Compose(
           [
                UniformTemporalSubsample(transform_params["num_frames"]),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(video_means, video_stds),
                ShortSideScale(size=transform_params["side_size"]),
                CenterCropVideo(
                    crop_size=(transform_params["crop_size"], transform_params["crop_size"])
                )
            ]
           ),
         ),
      ]
    )
	
    #lip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/fps
    clip_duration = 2
    refresh_rate = clip_duration

    # Initialize an EncodedVideo helper class and load the video
    video = EncodedVideo.from_path(video_path)
    
    cm1 = np.array([0,0,0,0]) # confusion matrix test 1 [TP, FN, FP, TN]
    cm2 = np.array([0,0,0,0]) # confusion matrix test 2 [TP, FN, FP, TN]
    
    start_sec = 0
    end_sec = start_sec + clip_duration
    extract_time = True
    end_test1 = False
    NoneType = type(None)
    starter.record()
    while True:

        if start_sec < duration and (end_sec - start_sec) > 3*clip_duration/4:
            # Select the duration of the clip to load by specifying the start and end duration
            # Load the desired clip
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            
            # Apply a transform to normalize the video input
                        
            try:
                video_data = transform(video_data)
            except:
                break
            
            device = "cuda"
            inputs = video_data["video"]
            inputs = inputs.to(device)
            
            # Pass the input clip through the model
            preds = model(inputs[None, ...])
            
            # Get the predicted classes
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_class = preds.topk(k=1).indices[0]
            
            if extract_time:
                ender.record()
                extract_time = True
            
            # test
            if label == 1 and pred_class.item() == 1: # TP
                if not end_test1:
                    end_test1 = True
                    cm1[0] += 1
                cm2[0] += 1
            elif label == 1 and pred_class.item() == 0: # FN test 2
                cm2[1] += 1
            elif label == 0 and pred_class.item() == 1: # FP test 2
                cm2[2] += 1
            else: # TN
                if not end_test1:
                    end_test1 = True
                    cm1[3] += 1
                cm2[3] += 1

            # next clip
            start_sec += refresh_rate
            end_sec = start_sec + clip_duration
        else:
            break
    
    if not end_test1:
        if label == 1 and pred_class.item() == 0: # FN test 1
            cm1[1] += 1
        else: # FP test 1
            cm1[2] += 1 
            
    test_time = starter.elapsed_time(ender)

    return cm1, cm2, test_time
 
# main
def main():
    parser = argparse.ArgumentParser()
	
    # ckpt
    parser.add_argument("--model_type", default="x3d_xs", type=str)
    parser.add_argument("--ckpt_path", default="/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/lightning_logs/x3d_xs_bi/checkpoints/x3d_xs-epoch=79-val_loss=0.317.ckpt",
    type=str, help="Checkpoint file path (.ckpt)")
    parser.add_argument("--hparams_path", default="/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/lightning_logs/x3d_xs_bi/hparams.yaml",
    type=str, help="Hyperparameters file path(.yaml)")
    
    # input video
    parser.add_argument("--testfile_path", default ="/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/data/annotation/test_bi (all classes)/test_bi.csv",
    type=str, help="Videos file path(.mp4)")
    
    args = parser.parse_args()
    
    # main
    model = VideoClassificationLightningModule.load_from_checkpoint(args.ckpt_path, hparams_file=args.hparams_path) #loads model
    
    file1 = open(args.testfile_path)
    
    tt_list = []
    cm1 = np.array([0,0,0,0])
    cm2 = np.array([0,0,0,0])
    while True:
        # Get next line and parses it
        line = file1.readline()
        if line!='':
            elems = line.split()
            filepath = elems[0]
            label = elems[1] # 1:sidewalk, 0:no_sidewalk

            c1, c2, tt = inference(model, filepath, int(label), model_transform_params[args.model_type])
            cm1 += c1
            cm2 += c2
            if tt != 0:
                tt_list.append(tt)
                
		#if line it's empty, ends process
        if not line:
            break
            
    # Compute time's mean, variance and std
    tt_array = np.array(tt_list)
    tt_mean = tt_array.mean()
    tt_var = tt_array.var()
    tt_std = tt_array.std()
    
    print()
    print('RESULTS ------------------')
    print('Confusion matrix (test1):')
    print(cm1)
    print()
    print('Confusion matrix (test2):')
    print(cm2)
    print()
    print('average test time (ms): ' + str(tt_mean) + ' / variance = ' + str(tt_var) + ' / std = ' + str(tt_std))

if __name__ == "__main__":
    main()
		
    
        
