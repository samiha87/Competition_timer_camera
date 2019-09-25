# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import array as arr
import threading
import queue
from yolo_detect import YoloDetector

startFrame = None
endFrame = None

# Find frames
def movement_detect(video_path, video_type, rotate):
	object_treshold = 0.10
	first_run = 0
	cap = cv2.VideoCapture(video_path)
	x1 = 1
	x2 = 20
	y1 = 100
	y2 = 200
	captured_pixels = 0
	changed_pixels = 0
	detection_on = False
	frames_detected = 0
	detector_lock = False
	timer_running = False

	while True:
		(grabbed, frame) = cap.read()
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			video_path = "videos/end_sample.mp4"
			x1 = 1
			x2 = 20
			y1 = 100
			y2 = 200
			captured_pixels = 0
			changed_pixels = 0
			detection_on = False
			frames_detected = 0
			detector_lock = False
			first_run = 0
			cap = cv2.VideoCapture(video_path)
			continue
		if rotate:
			rows,cols = frame.shape[:2]

			M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
			frame = cv2.warpAffine(frame,M,(cols,rows))
			original = frame.copy()

		if first_run < 2:
			(h, w) = frame.shape[:2]
			video_width = w
			video_height = h
			pixel_memory = np.zeros((int(video_height), int(video_width), 3), np.uint8)
			first_run = first_run + 1
			x1 = int(w/2) - 10
			x2 = int(w/2) + 10
			y1 = 100
			y2 = int(h)

		if detection_on:
			for x in range(x1, x2, 5):
				for y in range(y1, y2, 5):
					for d in range(3) :
						dd = pixel_memory[y][x][d]
						dd2 = frame[y][x][d]
						if ((dd + 20) < dd2 ) or ((dd - 20) > dd2):
							changed_pixels = changed_pixels + 1

			changed_pixels = changed_pixels * 5*5
			#print("Changed pixels "+ str(changed_pixels))
			#print("captured pixels" + str(captured_pixels))
			#print("Division " + str(changed_pixels/captured_pixels))
			if(detector_lock and (time.time() > (detector_delay + 20))):
				detector_lock = False
				print("detector_lock released")

			if( ((changed_pixels/captured_pixels) > object_treshold) and (detector_lock is False) ):
				frames_detected = frames_detected + 1
				if(frames_detected > 2):
					if(timer_running):
						print("Person detected, Stopping timer")
						frames_detected = 0
						#Get clock
						detector_delay = time.time()
						detector_lock = True
						print("Captured time: " + str(time.time() - rider_time) + " seconds")
						timer_running = False
						cv2.waitKey(1000)
						return
					else:
						print("Person detected, Starting timer")
						frames_detected = 0
						#Get clock
						detector_delay = time.time()
						detector_lock = True
						rider_time = time.time()
						timer_running = True
			else:
				frames_detected = 0

		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3 )
		display = cv2.resize(frame, (int(w * 0.4),int(h * 0.4)))
		cv2.imshow('frame', display)
		#cv2.imshow('original', original)

		changed_pixels = 0
		k = cv2.waitKey(10) & 0xFF
		if k == ord('s'):
			# Store pixels from rectangle area
			if rotate == False:
				 (grabbed, original) = cap.read()
			(h, w) = original.shape[:2]
			x1 = int(w/2) - 10
			x2 = int(w/2) + 10
			y1 = 100
			y2 = int(h)
			for x in range(x1, x2):
				for y in range(y1, y2):
					for d in range(3) :
						pixel_memory[y][x][d] = original[y][x][d]
						captured_pixels = captured_pixels + 1
			print("Starting detection, captured pixels: " + str(captured_pixels))
			detection_on = True
		if k == 27:
			break

# Color detector
def color_detect(video_path, video_type):
	cap = cv2.VideoCapture(0)

	while True:
		(grabbed, frame) = cap.read()
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break

		video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		#Convert images from BGR to HSV
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		lower_red = np.array([110, 50, 50])
		upper_red = np.array([130, 255, 255])

		# Here we are defining range of bluecolor in hsv
		# This creates a mask of blue coloured
		# Objects found in the frame
		mask = cv2.inRange(hsv, lower_red, upper_red)

		# The bitwise and of the frame and mask is done so
		# that only the blue coloured objects are highlighted
		# and stored in res
		res = cv2.bitwise_and(frame,frame, mask= mask)
		cv2.line(frame, (320, 1), (320, int(video_height)), (0, 255, 0), 2)
		cv2.imshow('frame',frame)
		cv2.imshow('mask',mask)
		cv2.imshow('res',res)
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break
# Yolo

def getLatestStartFrame():
	return

def getLatestEndFrame():
	return

# Start reading from 2 sources
def yolo_detect_both(video_path_start, video_path_end, video_type, yolo, rotate, shift, confidence_lim, treshold_lim, frame_rate):
	# Initialize Queue
	print("yolo_detect_both() Start source: " + video_path_start)
	print("yolo_detect_both() End source: " + video_path_end)
	if video_type != "both":
		print("yolo_detect_both() Wrong type " + video_type)
		return

	queue_start = queue.Queue()
	queue_end = queue.Queue()

	# Create a lock
	lock = threading.Lock()
	# Create threads, First queue input is to transmit, second to read
	startThread = YoloDetector(lock, video_path_start, "start", yolo, queue_end, queue_start, rotate, shift, confidence_lim, treshold_lim, frame_rate)
	endThread	= YoloDetector(lock, video_path_end, "end", yolo, queue_start, queue_end, rotate, shift, confidence_lim, treshold_lim, frame_rate)
	# Start monitoring end gate
	startThread.start()
	time.sleep(10)
	# Start monitoring start gate
	endThread.start()
	# Join queue
	queue_start.join()
	queue_end.join()

def translateImage(image, offsetx, offsety):
	rows, cols = image.shape[:2]
	M = np.float32([[1,0,offsetx], [0,1,offsety]])
	dst = cv2.warpAffine(image, M, (cols,rows))
	return dst

def yolo_detect(video_path, video_type, yolo, rotate, shift, confidence_lim, treshold_lim, frame_rate):
# load the COCO class labels our YOLO model was trained on
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
	print("Starting detection " + video_type)
	while True:
		
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		
		frame_count = frame_count + 1
		print("Frame count: " + str(frame_count))
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
			frame = cv2.warpAffine(frame, M, (H, W))

			#Translate image 
			frame = translateImage(frame, shift[0], shift[1])

			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
			net.setInput(blob)

			layerOutputs = net.forward(ln)
			end = time.time()

			# initialize our lists of detected bounding boxes, confidences,
			# and class IDs, respectively
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
					if confidence > confidence_lim:
						# scale the bounding box coordinates back relative to
						# the size of the image, keeping in mind that YOLO
						# actually returns the center (x, y)-coordinates of
						# the bounding box followed by the boxes' width and
						# height
					
						(Hd, Wd) = frame.shape[:2]
						box = detection[0:4] * np.array([Wd, Hd, Wd, Hd])
						(centerXd, centerYd, widthd, heightd) = box.astype("int")
						# use the center (x, y)-coordinates to derive the top
						# and and left corner of the bounding box
						x = int(centerXd - (widthd / 2))
						y = int(centerYd - (heightd / 2))

						# update our list of bounding box coordinates,
						# confidences, and class IDs
						boxes.append([x, y, int(widthd), int(heightd)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			# apply non-maxima suppression to suppress weak, overlapping
			# bounding boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_lim, treshold_lim)
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
					text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
					print(text)
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

					# Detect if person collides wit start line
		# get Heigh and width
		(H_, W_) = frame.shape[:2]	
		# Draw line middle of frame
		if frame_count >= frame_rate:
			cv2.line(frame, (W_/2, 1), (W_/2, H_), (0, 255, 0), 2)
			cv2.imshow(video_type, frame)
			cv2.waitKey(1)
			frame_count = 0
		# release the file pointers
	print("[INFO] cleaning up...")
	writer.release()
	vs.release()
