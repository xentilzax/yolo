import cv2
import argparse
import time
import numpy as np

class Model:
    def __init__(
            self,
            netSize: (int, int),
            config: str,
            weights: str,
            threshold: float,
            backend = cv2.dnn.DNN_BACKEND_CUDA,
            target = cv2.dnn.DNN_TARGET_CUDA_FP16):
        
        self._netSize = netSize
        self._config = config
        self._weights = weights
        self._threshold = threshold
        self._thresholdNMS = 0.5
        self._net = cv2.dnn.readNetFromDarknet(config, weights)
        self._net.setPreferableBackend(backend)
        self._net.setPreferableTarget(target)
        self._timeProcess = -1
        # determine only the *output* layer names that we need from YOLO
        self._ln = self._net.getLayerNames()
        self._ln = [self._ln[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]
        self._W = 0
        self._H = 0

    """
    prepare image for forward
    """
    def preProcessing(self, image):
        (self._H, self._W) = image.shape[:2]        
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, self._netSize, swapRB=True, crop=False)        
        self._net.setInput(blob)

    """
    analyze output, check theshold confidence and NMS, remove dublicate of boxes
    """
    def postProcessing(self, layerOutputs):
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
                if confidence > self._threshold:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([self._W, self._H, self._W, self._H])
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
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self._threshold, self._thresholdNMS)

        result = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                element = {}
                element["class"] = classIDs[i] 
                element["confidence"] = confidences[i]
                # extract the bounding box coordinates
                (element["x"], element["y"]) = (boxes[i][0], boxes[i][1])
                (element["w"], element["h"]) = (boxes[i][2], boxes[i][3])
                result.append(element)
                
        return result

    """
    Find objects into the image
    Return: boxes: list of dict. where dict:
    class: number of class
    confidence: float [0...1]
    x: coord in pixels
    y
    w
    h
    """
    def predict(self, image):
        start = time.time()
        
        self.preProcessing(image)
        layerOutputs = self._net.forward(self._ln)
        boxes = self.postProcessing(layerOutputs)
        
        end = time.time()        
        self._timeProcess = end-start
        
        return boxes

    def getTimeProcess(self):
        return self._timeProcess
