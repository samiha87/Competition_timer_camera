# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import threading
from detectors import yolo_detect, yolo_detect_both, color_detect, movement_detect

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", 		required=False, 	help="Start or end gate or both")
# Select input -> select webcam or video source
ap.add_argument("-s", "--inputstart", 	required=True, 		help="Select input for start gate")
ap.add_argument("-e", "--inputend",		required=False,		help="Select input for end gate")
ap.add_argument("-t", "--type", 		required=True, 		help="Set detection type")
ap.add_argument("-r", "--rotate", 		type=int, 			default=0,								required=False, 	help="Rotate image")
ap.add_argument("-a", "--translate",	nargs='+',			required=False,							help="Translate is used to shift image")
ap.add_argument("-y", "--yolo",			required=False,		help="Set directory for yolo data")
ap.add_argument("-c", "--confidence", 	type=float, 		default=0.5,							help="minimum probability to filter weak detections")
ap.add_argument("-l", "--treshold", 	type=float, 		default=0.3,							help="threshold when applyong non-maxima suppression")
ap.add_argument("-f", "--framerate", 	type=int,	 		default=5,								help="Frame rate to process images")
args = vars(ap.parse_args())


if args["type"] == "yolo":

	if args["mode"] == "both":
		#start yolo detector for start gate in thread
		yolo_detect_both(args["inputstart"], args["inputend"], args["mode"], args["yolo"], args["rotate"], args["translate"], args["confidence"], args["treshold"], args["framerate"])

	if args["mode"] == "start":
		yolo_detect(args["inputstart"], args["mode"], args["yolo"], args["rotate"], args["translate"], args["confidence"], args["treshold"], args["framerate"]) 

if args["type"] == "color":
	color_detect(args["input"], args["mode"])

if args["type"] == "movement":
	if args["rotate"] == "true":
		movement_detect(args["inputstart"], args["mode"], True)
	else :
		movement_detect(args["inputstart"], args["mode"], False)
