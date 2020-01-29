import face_alignment
from skimage import io
import argparse
import sys, os
from sklearn.cluster import DBSCAN
import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Face alignment')
parser.add_argument('--im', type=bool, default=False, help='True if testing on single image or False for directory')
parser.add_argument('--im_dir', type=str, default='', help='Image directory')
args = parser.parse_args()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cuda')

means = []
files = os.listdir(args.im_dir)
files.sort()
num = len(files)

if args.im:
    input = io.imread(args.im_dir)
    preds = fa.get_landmarks_from_image(input)
else:
    for i, f in enumerate(files):
        preds = fa.get_landmarks_from_image(args.im_dir + f)
        x = [item[0] for item in preds[0]]
        x = sum(x) / len(x)
        y = [item[1] for item in preds[0]]
        y = sum(y) / len(y)
        z = [item[2] for item in preds[0]]
        z = sum(z) / len(z)
        means.append([x,y,z])
        if i % 100 == 0:
            print('{0:.2f}% of images have been processed...'.format(i / num * 100))

print('Alignment Done')
print(len(means))

means = np.array(means)
clustering = DBSCAN(eps=2, min_samples=3).fit(X)
print(clustering.labels_)

index = [i for i, e in enumerate(clustering.labels_) if e == -1]

with open('./Outliers.txt', 'w') as f:
    for elem in index:
        f.write(files[elem])
        f.write('\n')
    f.close()

