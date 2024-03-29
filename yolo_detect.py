
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import array as arr
from multiprocessing  import Process, Queue, Lock
import sys

class YoloDetector (Process):
    def __init__(self, lock, path, vType, yolo, qin, qout, rotate, shift, confidence_lim, treshold_lim, frame_rate):
        Process.__init__(self)
        self.lock = lock
        self.path = path
        self.vType = vType
        self.qin = qin
        self.qout = qout
        self.rotation = rotate
        self.yolo = yolo
        self.shift = shift
        self.confidence_lim = confidence_lim
        self.treshold_lim = treshold_lim
        self.frame_rate = frame_rate

    def run(self):
        yolo_detect_start(self, self.path, self.vType, self.yolo, self.qin, self.qout, self.rotation, self.shift, self.confidence_lim, self.treshold_lim, self.frame_rate)
        self.is_alive = False
        sys.exit()

def translateImage(image, offsetx, offsety):
	rows, cols = image.shape[:2]
	M = np.float32([[1,0,offsetx], [0,1,offsety]])
	dst = cv2.warpAffine(image, M, (cols,rows))
	return dst

def storeFile(image, path) :
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

def yolo_detect_start(self, video_path, video_type, yolo, queue_input, queue_output, rotate, shift, confidence_lim, treshold_lim, frame_rate):
	# initialize a list of colors to represent each possible class label
    cv2.ocl.setUseOpenCL(False)
    np.random.seed(42)
    loopTimeStart = time.time() 
	# derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo, "yolov3.weights"])
    configPath = os.path.sep.join([yolo, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
	# and determine only the *output* layer names that we need from YOLO
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# initialize the video stream, pointer to output video file, and
	# frame dimensions
    vs = cv2.VideoCapture(video_path)	# Read video
    # Sizes of video stays stable we only need to pull them once
    # When pulling Frame and video width from GPU this causes slow down
    # OpenCV has to convert the gpu data to cpu data and this causes slow down
    # We change these values only if needed
    # Doing this increased performance majorily
    writer = None
    (W, H) = (None, None)
    (W2, H2) = (None, None)
    (W3, H3) = (None, None)

    Detection_count = 1
    Process_frame = frame_rate
    frame_count = 0

    block_gate = False
    timer_running = False
    # Init timers
    start = 0

    while True:
        
        #check queue if we have message from another thread
        try:
            thread_data = queue_input.get(False)
            if thread_data:
                # Queue requires a string format, convert back from string to float
                start = float(thread_data)  
                timer_running = True
        except:
            thread_data = None

        # read the next frame from the file
        (grabbed, frame) = vs.read()
        #try:
        #    frameGPU = cv2.UMat(frame) # Convert to OpenCL format to process through GPU
        #except:
        #   continue
        
        frame_count = frame_count + 1
        if(frame_count >= Process_frame):
        	# if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break

        	# if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            #Rotate image
            center = (W / 2, H /2)
            M = cv2.getRotationMatrix2D(center, float(rotate), 1)
            # Not supported on cuda
            frame = cv2.warpAffine(frame, M, (H, W))
            #Translate image 
            frame = translateImage(frame, shift[0], shift[1])
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)

            layerOutputs = net.forward(ln)

            # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            counter = 0
			# loop over each of the layer outputs, Can this be refactored
            # Looping thourgh like this is very slow
            for output in layerOutputs:
                # loop over each of the detections
                #print("Output layer size " + str(len(output)))
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
				    # filter out weak predictions by ensuring the detected
				    # probability is greater than the treshold probability
                    counter = counter +1
                    if confidence > confidence_lim:
                        if W2 is None or H2 is None:
                            (H2, W2) = frame.shape[:2]
                        box = detection[0:4] * np.array([W2, H2, W2, H2])
                        (centerXd, centerYd, widthd, heightd) = box.astype("int")
                        x = int(centerXd - (widthd / 2))
                        y = int(centerYd - (heightd / 2))

                        boxes.append([x, y, int(widthd), int(heightd)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        break # Break because we don't need to loop over rest.
			
            # apply non-maxima suppression to suppress weak, overlapping
		    # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_lim, treshold_lim)
		    # ensure at least one detection exists
            if (len(idxs) > 0 and (block_gate == False)):
				# loop over the indexes we are keeping
                for i in idxs.flatten():
				    # extract the bounding box coordinates
                    # Do this only once
                    if W3 is None or H3 is None:
                        (H3, W3) = frame.shape[:2]
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    Detection_count = Detection_count + 1
                    if (x < W3/2) and ( (x + w) > W3/2):
                        if(video_type is "start") :
                            print("Person crossed start ", Detection_count)
                            # Send start time to end gate
                            queue_output.put(str(time.time()))

                        if video_type is "end":
                            print("Timer stopped")
                            if timer_running is True:
                                stop = time.time()
                                elap = (stop - start)
                                print("Time elapsed {:.4f} seconds".format(elap))
                                timer_running = False

        if frame_count >= frame_rate:
            frame_count = 0
	# release the file pointers
    if writer is not None:
        writer.release()
    vs.release()
    elapsedTime = time.time() - loopTimeStart 
    print("\nProcess took = " + str(elapsedTime) + " " + video_type)