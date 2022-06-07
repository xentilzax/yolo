import cv2
import argparse
import time
import numpy as np


"""
prepare image for forward
"""
def preProcessing(image, net, szNet):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, szNet, swapRB=True, crop=False)
    
    net.setInput(blob)

"""
analyze output, check theshold confidence and NMS, remove dublicate of boxes
"""
def postProcessing(layerOutputs, thConfidence, thNMS):
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > thConfidence:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)


    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, thConfidence, thNMS)

    result = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            element = {}
            element["class"] = classIDs[i] 
            # extract the bounding box coordinates
            (element["x"], element["y"]) = (boxes[i][0], boxes[i][1])
            (element["w"], element["h"]) = (boxes[i][2], boxes[i][3])
            result.append(element)
            
    return result

"""
draw boundary boxes of objects on image
"""
def drawBoxes(frame, boxes, color):
    for box in boxes:
        if box["class"] != 0:
            continue
        
        (x, y) = (box["x"], box["y"])
        (w, h) = (box["w"], box["h"])

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
    return image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to the video file")
ap.add_argument("-o", "--output", default="predict.jpg", help="set output video name of file")
ap.add_argument("-t", "--confidence", default=0.5, help="theshold confidence")
ap.add_argument("-n", "--nms", default=0.5, help="NMS theshold")
args = vars(ap.parse_args())

configPath = "football.cfg"
weightsPath = "football.weights"
netW=800
netH=608

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

image = cv2.imread(args["input"])
(H, W) = image.shape[:2]
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

preProcessing(image, net, (netW, netH))

start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
print("time: {} sec".format(end-start))

boxes = postProcessing(layerOutputs, args["confidence"], args["nms"])

# draw a bounding box rectangle and label on the image
image = drawBoxes(image, boxes, (0, 255, 255))

# show the output image
cv2.imwrite(args["output"], image)
