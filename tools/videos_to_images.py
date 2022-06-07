"""
parse input video to images .png files
example run:
python3 video_to_images.py -l=list.csv

file list.csv contain list of video files:

/home/entilza/work/test_task/data/video1.webm
/home/entilza/work/test_task/data/video4.mp4
/home/entilza/work/test_task/data/video2.mp4
/home/entilza/work/test_task/data/video3.mp4

rate by parse image = 2 (2 images per second)
parsed images saving into dir:
./images

"""
import pandas as pd
import argparse
import os
import subprocess

videos = None

def func():
    return


ap = argparse.ArgumentParser()
ap.add_argument("-l", "--list", type=str, default="list.csv", help="list file images")

args = vars(ap.parse_args())
listFile = args["list"]

with open(listFile) as file:
    for line in file:
        video_filename = line.rstrip()
        #video_filename = "/home/entilza/work/data/tikkurila/real-video/sunny-1.mp4"
        title, ext = os.path.splitext(os.path.basename(video_filename))        
        print(video_filename)
        
        subprocess.call("ffmpeg -i \"{}\" -r 2/1 \"images/{}-%03d.png\""
                       .format(video_filename, title), shell=True)
