import os
import cv2

def save_vids_as_img(filepath, save_directory, binary:bool):
    file1 = open(filepath)
    count = 0
    
    while True:
        count += 1

        # Get next line and parses it
        line = file1.readline()
        if line!='':
            elems = line.split()
            fn = elems[0] #./data/video_data/BikeU/BikeU78.mp4
            
            #extracts where to save
            foldername = fn.split('/')[-2]
            if binary and foldername != 'sidewalk':
                foldername = 'other'
            save_path = os.path.join(save_directory, foldername)
            
            #saves video as a set of images
            vidcap = cv2.VideoCapture(fn)
            
            success,image = vidcap.read()
            count = 0
            while success:
                img_name = fn.split('/')[-1] + str(count) + '.jpg'
                img_name = os.path.join(save_path, img_name)
                cv2.imwrite(img_name, image) # save frame as JPEG file      
                success,image = vidcap.read()
                count += 1
		
		#if line it's empty, ends process
        if not line:
            break

def main():
    reference_path = '/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/data/video_data'
    image_path = '/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/data/image_data'
    train_files = '/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/data/annotation/train.csv'
    val_files = '/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/data/annotation/val.csv'
    test_files = '/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/data/annotation/test.csv'
    test_files_only = True
    binary = False
	
    if binary:
        print('binary mode')

	    #replicates reference_path for train, val and test
        dir_list = os.listdir(reference_path)
		
        binary_dir_list = ['sidewalk', 'other']      
        for subclass in binary_dir_list:
            if  not test_files_only:
                newpath = os.path.join(image_path, 'train_bi', subclass)
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                newpath = os.path.join(image_path, 'val_bi', subclass)
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
            newpath = os.path.join(image_path, 'test_bi', subclass)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
                
        if not test_files_only:
            # saves train videos as images
            print('creating train images...')
            train_path = os.path.join(image_path, 'train_bi')
            save_vids_as_img(train_files, train_path, binary)
            # saves val videos as images
            print('creating val images...')
            val_path = os.path.join(image_path, 'val_bi')
            save_vids_as_img(val_files, val_path, binary)
        # saves test videos as images
        print('creating test images...')
        test_path = os.path.join(image_path, 'test_bi')
        save_vids_as_img(test_files, test_path, binary)
        
    else:
		
	    #replicates reference_path for train, val and test
        dir_list = os.listdir(reference_path)
		      
        for subclass in dir_list:
            if  not test_files_only:
                newpath = os.path.join(image_path, 'train', subclass)
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                newpath = os.path.join(image_path, 'val', subclass)
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
            newpath = os.path.join(image_path, 'test', subclass)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
                
        if not test_files_only:
            # saves train videos as images
            train_path = os.path.join(image_path, 'train')
            save_vids_as_img(train_files, train_path, binary)
            # saves val videos as images
            val_path = os.path.join(image_path, 'val')
            save_vids_as_img(val_files, val_path, binary)
        # saves test videos as images
        test_path = os.path.join(image_path, 'test')
        save_vids_as_img(test_files, test_path, binary)
    
if __name__ == "__main__":
    main()
