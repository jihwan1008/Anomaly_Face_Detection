import face_alignment
from skimage import io
import argparse

parser = argparse.ArgumentParser(description='Face alignment')
parser.add_argument('--im', type=bool, default=True, help='True if testing on single image or False for directory')
parser.add_argument('--im_dir', type=str, default='', help='Image directory')
args = parser.parse_args()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

input = io.imread(args.im_dir)
preds = fa.get_landmarks(input)
print(preds)
