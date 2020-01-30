import face_recognition as fr
import argparse
import sys, os
import numpy as np
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser(description='face_recog')
parser.add_argument('--im_dir', type=str, help='location of image folder')
parser.add_argument('--thres', type=float, help='threshold for face distance')
args = parser.parse_args()

files = os.listdir(args.im_dir)
files.sort()

default = str
for elem in files:
    if elem[-11:-4] == 'default':
        default = args.im_dir + elem
print('Base image : {}'.format(default))

known_image = fr.load_image_file(default)
known_encoding = fr.face_encodings(known_image)[0]

encode_list = []

for item in tqdm(files):
    if item[-11:-4] == 'default':
        pass
    else:
        item = fr.load_image_file(args.im_dir + item)
        encode_list.append(fr.face_encodings(item)[0])

distance = fr.face_distance(encode_list, known_encoding)
print(distance)

index = [i for i, e in enumerate(distance) if float(e) > args.thres]
index_ = [i for i, e in enumerate(distance) if float(e) < args.thres]

f_out = './Filtered_Out_' + str(args.thres) + '/'
f_in = './Filtered_In_' + str(args.thres) + '/'

if os.path.isdir(f_out):
    pass
else:
    os.mkdir(f_out)

if os.path.isdir(f_in):
    pass
else:
    os.mkdir(f_in)

for elem in index:
    src = args.im_dir + files[elem]
    dst = f_out
    shutil.copy(src, dst)

for elem in index_:
    src = args.im_dir + files[elem]
    dst = f_in
    shutil.copy(src, dst)
