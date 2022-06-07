import cv2
import argparse
import sys
from model import Model 

"""
draw boundary boxes of objects on image
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
ap.add_argument("-t", "--confidence", default=0.5, help="theshold confidence")
ap.add_argument("-s", "--size", default="800x608", help="size output video")

args = vars(ap.parse_args())

#pring HELP if no args
if len(sys.argv)==1:
    ap.print_help(sys.stderr)
    sys.exit(1)
    
#CNN config
configPath = "football.cfg"
weightsPath = "football.weights"
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

def MainLoop():
    global image
    res, image = vs.read()
    if res == False or image is None:
        print('Process complete!')
        return False


    boxes = model.predict(image)

    # draw a bounding box rectangle and label on the image
    image = drawBoxes(image, boxes, (0, 255, 255))

    if fOutput == True and outputVideo != None:
        resized = cv2.resize(image, sizeOutput)
        outputVideo.write(resized)    
            
    return True

fPause = False
numFrames = 0

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

    dt = model.getTimeProcess()
    numFrames += 1    
    print("number frame complete: {}, fps: {}".format(numFrames, 1/dt), end = '\r')

