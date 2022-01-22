import os
import cv2
import random

def save_single_frame_as_img(filepath, save_directory, binary:bool):
    file1 = open(filepath)
    count = 0
    
    while True:
        count += 1

        # Get next line and parses it
        line = file1.readline()
        if line!='':
            elems = line.split()
            fn = elems[0] #./data/video_data/BikeU/BikeU78.mp4
            
            # extracts where to save
            foldername = fn.split('/')[-2]
            if binary and foldername != 'sidewalk':
                foldername = os.path.join(save_directory,'other') 
            save_path = os.path.join(save_directory, foldername)
            
            # extract a single random frame from the video
            cap = cv2.VideoCapture(fn)
                   
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        
            fr = random.randint(0, frame_count-1)

            img_name = fn.split('/')[-1] + str(fr) + '.jpg'                    
            img_name = os.path.join(save_path, img_name)
                        
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ret, frame = cap.read()
            
            cv2.imwrite(img_name, frame) # save frame as JPEG file

        print(fn + ' has been saved successfuly')
		
		#if line it's empty, ends process
        if not line:
            break

def main():
    reference_path = '/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/data/video_data'
    image_path = '/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/data/image_data'
    files = '/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/data/annotation/test.csv'
    binary = False
    
    if binary:
        dir_list = ['sidewalk', 'other']
    else:    
        #replicates reference_path for train and val
        dir_list = os.listdir(reference_path)

    for subclass in dir_list:
        if binary:
            newpath_name = 'test_single_frame_bi'
        else:
            newpath_name = 'test_single_frame' 			
        newpath = os.path.join(image_path,newpath_name, subclass)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            
    # saves test videos as images
    path = os.path.join(image_path, newpath_name)
    save_single_frame_as_img(files, path, binary)
    
if __name__ == "__main__":
    main()
