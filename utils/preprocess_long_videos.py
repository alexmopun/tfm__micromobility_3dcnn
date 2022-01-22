import os
import argparse
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def extract_filenames(origin_path, valid_subdirs):
	long_filenames = []
	
	for path, subdirs, files in os.walk(origin_path):
		for subdir in subdirs:
			if subdir in valid_subdirs:
				subdirpath = path + '/' + subdir + '/'
				video_list = os.listdir(subdirpath)
				
				for video_name in video_list:
					if 'PXL' in video_name and '.mp4' in video_name:
						video_path = os.path.join(subdirpath, video_name)
						long_filenames.append(video_path)
						
	return long_filenames		

def preprocess_video(video_path, destination_path):	
	# split video in clips
	try:
		clip = VideoFileClip(video_path)
		video_duration = clip.duration
		
		# extracts destination video_name and folder
		video_path_split = video_path.split('/')
		video_save_path = os.path.join(destination_path, video_path_split[-2], video_path_split[-1])
		if not os.path.exists(video_save_path):
			os.makedirs(video_save_path)
		
		start_sec = 0
		while (start_sec + 2) < video_duration:
			video_name = video_path_split[-1] + str(start_sec) + '.mp4'
			ffmpeg_extract_subclip(video_path, start_sec, start_sec+2, targetname=os.path.join(video_save_path, video_name))
			start_sec += 2
	except:
		print("error in video " + video_path)
# main
def main():
	# parser
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--origin_path", default="/mnt/gpid08/datasets/micromobility/lane_classification/videos", type=str)
	parser.add_argument("--destination_path", default="/mnt/gpid07/imatge/alex.moreno.punzano/Desktop/C3D/data/long_videos", type=str)
	
	args = parser.parse_args()
    
    # extracts long videos filenames (PXL in the name)
	valid_subdirs = ['BikeBi', 'BikeU', 'road', 'sidewalk']
	filenames = extract_filenames(args.origin_path, valid_subdirs)
	
	# generates new directories
	for subdir in valid_subdirs:
		newpath = os.path.join(args.destination_path, subdir)
		if not os.path.exists(newpath):
			os.makedirs(newpath)
	
	print('Pre-processing ' + str(len(filenames)) + ' videos, please be patient...' )
	
	# divides and saves long videos in 2 second clips
	for video in filenames:
		preprocess_video(video, args.destination_path)		

if __name__ == "__main__":
	main()
