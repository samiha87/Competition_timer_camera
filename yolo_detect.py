import numpy as np
import argparse
import imutils
import time
import cv2
import os
import array as arr
import threading
from Queue import Queue

class YoloDetector (threading.Thread):
    def __init__(self, lock, path, vType, qin, qout, rotate):
        threading.Thread.__init__(self)
        self.lock = lock
        self.path = path
        self.vType = vType
        self.qin = qin
        self.qout = qout
        self.rotation = rotate

    def run(self):
        yolo_detect_start(self, self.path, self.vType, self.qin, self.qout)

def yolo_detect_start(self, video_path, video_type, queue_input, queue_output):
    # load the COCO class labels our YOLO model was trained o
    def_confidence = 2
    def_threshold = 2
    labelsPath = os.path.sep.join(["yolo_files", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
	# initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

	# derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(["yolo_files/", "yolov3.weights"])
    configPath = os.path.sep.join(["yolo_files", "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
	# and determine only the *output* layer names that we need from YOLO
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	    # initialize the video stream, pointer to output video file, and
	    # frame dimensions
    print("Reading video file " + video_type)
    vs = cv2.VideoCapture(video_path)	# Read video
    print("Reading video file read" + video_type)
	    # Get dimensions from video start footage
    video_width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)

    writer = None
    (W, H) = (None, None)

    Detection_count = 1
    Process_frame = 10
    frame_count = 0

    block_gate = False
    timer_running = False
    print("Starting detection " + video_type)
    while True:
            # read the next frame from the file
        (grabbed, frame) = vs.read()
        frame_count = frame_count + 1
        if(frame_count >= Process_frame):
        	# if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break

        	# if the frame dimensions are empty, grab them
            if W is None or H is None:
    		    (H, W) = frame.shape[:2]
            
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)

            layerOutputs = net.forward(ln)
            end = time.time()
            # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

			# loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
				    # filter out weak predictions by ensuring the detected
				    # probability is greater than the minimum probability
                    if confidence > def_confidence:
                        # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
					    # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

					    # update our list of bounding box coordinateframe_counts, confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
			# apply non-maxima suppression to suppress weak, overlapping
		    # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, def_confidence, def_threshold)
            if(block_gate and ( (time.time() - block_gate_timer) > 5)) :
                block_gate = False
		    # ensure at least one detection exists
            if (len(idxs) > 0 and (block_gate == False)):
				# loop over the indexes we are keeping
                for i in idxs.flatten():
				    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    Detection_count = Detection_count + 1
                    if (x < W/2) and ( (x + w) > W/2):
                        if(timer_running == False) :
                            print("Person crossed start ", Detection_count)
                            start = time.time()
                            block_gate = True
                            block_gate_timer = time.time() 
                            print("Timer started")
                            timer_running = True
                        else:
                            timer_running = False
                            print("Timer stopped")
                            stop = time.time()
                            elap = (stop - start)
                            print("Time elapsed {:.4f} seconds".format(elap))
                            block_gate = True
                            block_gate_timer = time.time()
                    print("")
        print("Processing fame "+ str(frame_count) + " for " + video_type)
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()