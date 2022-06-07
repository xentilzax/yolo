"""
Show labeling on video.
example run:
python3 show_labeling -l=annotations1.xml -v=video1.webm

"""

import xml.etree.ElementTree as ET
import argparse
import cv2

def get_labels(in_file):
    print(in_file)
    tree=ET.parse(in_file)

    root = tree.getroot()
    meta = root.find('meta')
    task = meta.find('task')
    size = task.find('original_size')

    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    track = root.find('track')
    
    result = []
    
    for child in track:
        result.append(child.attrib)
        #child.find('frame')
        #print(child.tag, child.attrib)
    
    return result
    


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to video file")
ap.add_argument("-l", "--labels", type=str, help="path to annotation file")
args = vars(ap.parse_args())

labels = get_labels(args["labels"])
print("labels number loaded: ", len(labels))

video = cv2.VideoCapture(args["video"])
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print("frames count in video: ", frames_count)

i = 0
j = 0
fPause = False
while True:
    if fPause == False:
        res, frame = video.read()
        if res == False or frame is None:
            print('Video end')
            break
    
        #while True:
        label = labels[i]
            #if int(label['keyframe']) == 0:
                #break
            
        print(label)
        
        tl = (int(float(label["xtl"])), int(float(label["ytl"])))
        br = (int(float(label["xbr"])), int(float(label["ybr"])))
        cv2.rectangle(frame, tl, br, (255,255,0), 2)
        #cv2.rectangle(frame, (0, 0), (100, 100), (255,0,0), 2)
        s = "video frame {} : label frame {}".format(i, label["frame"])
        cv2.putText(frame, s, (0, int(frame.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("src", frame)

        i = i + 1
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("p"):
        fPause = not fPause
	# if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break
    
