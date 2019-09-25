import numpy as np
import argparse
import imutils
import time
import cv2
import os
import array as arr
import threading
import queue

class YoloDetector (threading.Thread):
    def __init__(self, lock, path, vType, yolo, qin, qout, rotate, shift, confidence_lim, treshold_lim, frame_rate):
        threading.Thread.__init__(self)
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

def translateImage(image, offsetx, offsety):
	rows, cols = image.get().shape[:2]
	M = np.float32([[1,0,offsetx], [0,1,offsety]])
	dst = cv2.warpAffine(image, M, (cols,rows))
	return dst

def yolo_detect_start(self, video_path, video_type, yolo, queue_input, queue_output, rotate, shift, confidence_lim, treshold_lim, frame_rate):
    # load the COCO class labels our YOLO model was trained o
    print(video_type + " Treshold " + str(confidence_lim))
    labelsPath = os.path.sep.join([yolo, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
	# initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

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
    print("Reading video file " + video_type)
    vs = cv2.VideoCapture(video_path)	# Read video
    print("Reading video file read" + video_type)
	    # Get dimensions from video start footage
    video_width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)

    writer = None
    (W, H) = (None, None)

    Detection_count = 1
    Process_frame = frame_rate
    frame_count = 0

    block_gate = False
    timer_running = False
    # Init timers
    start = 0
    end = 0

    print("Starting detection " + video_type)
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
        frameGPU = cv2.UMat(frame) # Convert to OpenCL format to process through GPU
        frame_count = frame_count + 1
        if(frame_count >= Process_frame):
        	# if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break

        	# if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frameGPU.get().shape[:2]
            #Rotate image
            center = (W / 2, H /2)
            M = cv2.getRotationMatrix2D(center, float(rotate), 1)
            # Not supported on cuda
            frameGPU = cv2.warpAffine(frameGPU, M, (H, W))

            #Translate image 
            frameGPU = translateImage(frameGPU, shift[0], shift[1])
            
            blob = cv2.dnn.blobFromImage(frameGPU, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)

            layerOutputs = net.forward(ln)
            end = time.time()

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

                        (Hd, Wd) = frameGPU.get().shape[:2]
                        box = detection[0:4] * np.array([Wd, Hd, Wd, Hd])
                        (centerXd, centerYd, widthd, heightd) = box.astype("int")
                        x = int(centerXd - (widthd / 2))
                        y = int(centerYd - (heightd / 2))

                        boxes.append([x, y, int(widthd), int(heightd)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        break; # Break because we don't need to loop over rest.
			
            # apply non-maxima suppression to suppress weak, overlapping
		    # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_lim, treshold_lim)
            if(block_gate and ( (time.time() - block_gate_timer) > 5)):
                block_gate = False
		    # ensure at least one detection exists
            if (len(idxs) > 0 and (block_gate == False)):
				# loop over the indexes we are keeping
                for i in idxs.flatten():
				    # extract the bounding box coordinates
                    (H, W) = frameGPU.get().shape[:2]
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    Detection_count = Detection_count + 1
                    if (x < W/2) and ( (x + w) > W/2):
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
        # get Heigh and width
        #if frameGPU is not None:
        #    (H_, W_) = frameGPU.get().shape[:2]	
        # Draw line middle of frame
        if frame_count >= frame_rate:
            frame_count = 0
		# release the file pointers
    print("[INFO] cleaning up...")
    if writer is not None:
        writer.release()
    vs.release()