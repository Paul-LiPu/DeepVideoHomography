import cv2
import os
import argparse
from src import util

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help='the path to video files')
parser.add_argument('--output_path', type=str, required=True, help='the path to output directory')
parser.add_argument('--output_height', type=int, default=720, help='output image height')
parser.add_argument('--output_width', type=int, default=1280, help='output image width')
config = parser.parse_args()

video_dir = config.input_path
output_dir = config.output_path
video_files = util.globx(video_dir, ['*.MP4', '*.mp4', '*.mov'])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for video_file in video_files:
    vidcap = cv2.VideoCapture(video_file)
    print(video_file)
    video_name = os.path.basename(video_file)
    temp = video_name.split('.')
    video_name = temp[0]

    output_subdir = output_dir + '/' + video_name
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    success, image = vidcap.read()
    count = 0
    while success:
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
        # if count % 4 == 0:
        cv2.imwrite(output_subdir + '/'+ video_name + "_frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    vidcap.release()
