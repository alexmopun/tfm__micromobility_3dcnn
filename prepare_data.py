import os
import argparse

import random

def prepare_data():
	parser = argparse.ArgumentParser()
	
	# Paths
	parser.add_argument("--video_path", default = "./data/video_data/", type = str)
	parser.add_argument("--annotation_path", default = "./data/annotation/", type = str)

	# Split
	parser.add_argument("--ratio", default = (0.9,0.05,0.05), type=tuple)
	
	# Binary Mapping
	parser.add_argument("--binary", action='store_true')
	
	args = parser.parse_args()	
	checktuple(args.ratio)
		
	# Extracts classes from video_path, and assigns a label to each one
	v_root = args.video_path
	dir_list = os.listdir(v_root)
	
	lane_dict = dict()
	for key in range(len(dir_list)):
		if args.binary:
			if dir_list[key] == 'sidewalk':
				lane_dict[dir_list[key]] = 1
			else:
				lane_dict[dir_list[key]] = 0				
		else:
			lane_dict[dir_list[key]] = key
		
	print('	The following labels have been assigned : ' + str(lane_dict))
	
	# Generates classes file
	if args.binary:
		filepath = args.annotation_path + 'info_classes_bi.txt'
	else:
		filepath = args.annotation_path + 'info_classes.txt'
	f = open(filepath, 'w')
	for key in lane_dict:
		f.write(str(lane_dict[key]) + ' ' + key + '\n')
	f.close()	
	
	# Generates  and ordered list with video_dir and labels
	train_video_list = []
	val_video_list = []
	test_video_list = []
	
	train_ratio, val_ratio, test_ratio = args.ratio
	
	for path, subdirs, files in os.walk(v_root):
		for subdir in subdirs:
			subdirpath = path + subdir + '/'
			v_list = os.listdir(subdirpath)
			random.shuffle(v_list) # randomizes videos from directory
			v_train = v_list[:int((len(v_list)+1)*train_ratio)]
			v_val_test = v_list[int((len(v_list)+1)*train_ratio):]
			v_val = v_val_test[:int((len(v_val_test)+1)*(val_ratio/(val_ratio+test_ratio)))]
			v_test = v_val_test[int((len(v_val_test)+1)*(test_ratio/(val_ratio+test_ratio))):]
			
			for t in v_train:
				train_video_list.append((os.path.join(subdirpath, t), lane_dict[subdir]))
			
			for t in v_val:
				val_video_list.append((os.path.join(subdirpath, t), lane_dict[subdir]))
				
			for t in v_test:
				test_video_list.append((os.path.join(subdirpath, t), lane_dict[subdir]))	
	
	random.shuffle(train_video_list)
	random.shuffle(val_video_list)
	random.shuffle(test_video_list)
				
	# Generates the.csv for train and val splits
	if args.binary:
		filepath = args.annotation_path + 'train_bi.csv'
	else:
		filepath = args.annotation_path + 'train.csv'		
	f = open(filepath, 'w')
	for v in train_video_list:
		f.write(v[0] + ' ' + str(v[1]) + '\n')
	f.close()
	
	if args.binary:
		filepath = args.annotation_path + 'val_bi.csv'
	else:
		filepath = args.annotation_path + 'val.csv'
	f = open(filepath, 'w')
	for v in val_video_list:
		f.write(v[0] + ' ' + str(v[1]) + '\n')
	f.close()
	
	if args.binary:
		filepath = args.annotation_path + 'test_bi.csv'
	else:
		filepath = args.annotation_path + 'test.csv'
	f = open(filepath, 'w')
	for v in test_video_list:
		f.write(v[0] + ' ' + str(v[1]) + '\n')
	f.close()
	
	print('	.csv files have been generated correctly.')

def checktuple(in_tuple):
	if sum(list(in_tuple)) < 0.99 or sum(list(in_tuple)) > 1:
		raise argparse.ArgumentTypeError('train_ratio + val_ratio + test_ratio must be 1')
	
if __name__ == "__main__":
	prepare_data()
