import cv2
import argparse
import sys
from model import Model 
import pandas as pd

"""
drawing the boundaries of objects on the image
"""
def drawBoxes(image, boxes, color):
    for box in boxes:
        if box["class"] != 0:
            continue

        (x, y) = (box["x"], box["y"])
        (w, h) = (box["w"], box["h"])

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    return image



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to the video file")
ap.add_argument("-o", "--output", default="output.mp4", help="set output video name of file")
ap.add_argument("-v", "--verbose", help="set verbose mode, show debug info", action='store_true')
ap.add_argument("-c", "--config", default="football.cfg", help="config CNN")
ap.add_argument("-w", "--weights", default="football.weights", help="weights CNN")
ap.add_argument("-t", "--confidence", default=0.5, help="theshold confidence")
ap.add_argument("-s", "--size", help="size output video")
ap.add_argument("-l", "--labeling", help="path for saving annotation file")

args = vars(ap.parse_args())

#pring HELP if no args
if len(sys.argv)==1:
    ap.print_help(sys.stderr)
    sys.exit(1)

#CNN config
configPath = args["config"]
weightsPath = args["weights"]
netSize= (800, 608)
#Flags and variables
sizeOutput = (0,0)
fVideo = False
fOutput =False
fDebugMode = False
vs = None

if args.get("verbose", False) !=  False:
    fDebugMode = True

if args.get("input", None) !=  None:
    fVideo = True
    vs = cv2.VideoCapture(args["input"])
    fps = vs.get(cv2.CAP_PROP_FPS)
    print('Input video fps: ',fps)
    width  = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sizeOutput = (width, height)

if args.get("size", None) !=  None:
    sizeOutput = tuple([int(a) for a in args["size"].split("x")])
    print("Output video resolution: ",sizeOutput)

if args.get("output", None) !=  None:
    filename = args["output"]
    fOutput = True
    print('Write video to file: ', filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outputVideo = cv2.VideoWriter(filename, fourcc, fps, sizeOutput)

#create model
model = Model(netSize, configPath, weightsPath, 0.5)

image = None
columns=['frame_id', 'class', 'confidence', 'x','y','width','height']
df = pd.DataFrame(columns=columns)
#print(df)
fPause = False
numFrames = 0

def MainLoop():
    global image, df
    res, image = vs.read()
    if res == False or image is None:
        print('Process complete!')
        return False


    boxes = model.predict(image)

    for box in boxes:
        data = [numFrames, 0, box["confidence"], box["x"], box["y"], box["w"], box["h"]]
        df = pd.concat([df, pd.DataFrame([data], columns=columns)], ignore_index=True)

    # draw a bounding box rectangle and label on the image
    image = drawBoxes(image, boxes, (0, 255, 255))

    if fOutput == True and outputVideo != None:
        resized = cv2.resize(image, sizeOutput)
        outputVideo.write(resized)

    return True


while True:
    if fDebugMode == True:
        if fPause == False:

            if MainLoop() == False:
                break
            # show the output image
            cv2.imshow("output", image)

        key = cv2.waitKey(1) & 0xFF
        if key == 32: #space
            fPause = not fPause
        # if the Esc key is pressed, break from the loop
        if key == 27: #Esc
            break

    else:
        if MainLoop() == False:
            break

    total = model.getTimeProcess()
    (Tpre, Tf, Tpost) = model.getDetailTimeProcess()
    numFrames += 1

    print("number frame complete: {0}, "
          "time: {1:.3f} sec, [{2:.3f}, {3:.3f}, {4:.3f}]".format(numFrames,
                                              total,
                                              Tpre, Tf, Tpost))

if args.get("labeling", None)  != None:
    df.to_csv(args["labeling"], index=False)
