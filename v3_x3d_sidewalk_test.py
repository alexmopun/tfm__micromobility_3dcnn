import argparse
import torch
import numpy as np
import torch.nn.functional as F
import os

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

def inference(model, folder_path, label, model_transform_params):
	
    model = model.cuda()

    # for inference time test
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
	
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
    
    v_files = os.listdir(folder_path)
    
    cm1 = np.array([0,0,0,0]) # confusion matrix test 1 [TP, FP, FN, TN]
    cm2 = np.array([0,0,0,0]) # confusion matrix test 1 [TP, FP, FN, TN]
    
    n_error = 0
    
    end_exp_1 = False
    record_time = True
       
    for v in v_files:
        v_path = os.path.join(folder_path, v)
        video = EncodedVideo.from_path(v_path)
        
        video_data = video.get_clip(start_sec=0, end_sec=2)
		
        if record_time:
            starter.record()
         
        # Apply a transform to normalize the video input
        video_data = transform(video_data)
        
        device = "cuda"
        inputs = video_data["video"]
        inputs = inputs.to(device)
        
        # Pass the input clip through the model
        preds = model(inputs[None, ...])

        if record_time:
            ender.record()
            record_time = False
        
        # Get the predicted classes
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_class = preds.topk(k=1).indices[0]                     
        
        # test
        if label == 1 and pred_class.item() == 1: # TP
            if not end_exp_1:
                end_exp_1 = True
                cm1[0] += 1
            cm2[0] += 1
        elif label == 0 and pred_class.item() == 0: # TN
            if not end_exp_1:
                end_exp_1 = True
                cm1[3] += 1
            cm2[3] += 1
        elif label == 1 and pred_class.item() == 0: # FN exp2
            cm2[2] += 1
        elif label == 0 and pred_class.item() == 1: # FP exp2
            cm2[1] += 1
        else:
            n_error += 1
    
    if not end_exp_1:
        if label == 1 and pred_class.item() == 0: # FN exp1
            cm1[2] += 1
        elif label == 0 and pred_class.item() == 1: # FP exp1
            cm1[1] += 1
        else: 
            n_error += 1

    test_time = starter.elapsed_time(ender)
    torch.cuda.empty_cache()

    return cm1, cm2, test_time, n_error    
	
# main
def main():
    parser = argparse.ArgumentParser()
	
    # ckpt
    parser.add_argument("--model_type", default="x3d_m", type=str)
    parser.add_argument("--ckpt_path", default="/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/lightning_logs/x3d_m_bi/checkpoints/x3d_m-epoch=59-val_loss=0.323.ckpt",
    type=str, help="Checkpoint file path (.ckpt)")
    parser.add_argument("--hparams_path", default="/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/lightning_logs/x3d_m_bi/hparams.yaml",
    type=str, help="Hyperparameters file path(.yaml)")
    
    # input video
    parser.add_argument("--testfile_path", default ="/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/data/annotation/test_folders.txt",
    type=str, help="Videos file path(.mp4)")
    
    args = parser.parse_args()
    
    # main
    model = VideoClassificationLightningModule.load_from_checkpoint(args.ckpt_path, hparams_file=args.hparams_path) #loads model
    
    file1 = open(args.testfile_path)
    
    tt_list = []
    n_errors = 0
    cm1 = np.array([0,0,0,0])
    cm2 = np.array([0,0,0,0])
    while True:
        # Get next line and parses it
        line = file1.readline()
        if line!='':
            elems = line.split()
            folderpath = elems[0]
            label = elems[1] # 1:sidewalk, 0:no_sidewalk

            c1, c2, tt, err = inference(model, folderpath, int(label), model_transform_params[args.model_type])
            cm1 += c1
            cm2 += c2
            n_errors += err
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
    print('Confusion matrix (exp1):')
    print(cm1)
    print()
    print('Confusion matrix (exp2):')
    print(cm2)
    print()
    print('Error count: ' + str(n_errors))
    print()
    print('average test time (ms): ' + str(tt_mean) + ' / variance = ' + str(tt_var) + ' / std = ' + str(tt_std))

if __name__ == "__main__":
    main()

