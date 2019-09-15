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
ap.add_argument("-e", "--inputend",		required=True,		help="Select input for end gate")
ap.add_argument("-t", "--type", 		required=True, 		help="Set detection type")
ap.add_argument("-r", "--rotate", 		required=False, 	help="Rotate image")
ap.add_argument("-a", "--translate",	nargs='+',			required=False,						help="Translate is used to shift image")
args = vars(ap.parse_args())


if args["type"] == "yolo":

	if args["mode"] == "both":
		#start yolo detector for start gate in thread
		yolo_detect_both(args["inputstart"], args["inputend"], args["mode"], args["rotate"])

	if args["mode"] == "start":
		yolo_detect(args["inputstart"], args["mode"], args["rotate"], args["translate"]) 

if args["type"] == "color":
	color_detect(args["input"], args["mode"])

if args["type"] == "movement":
	if args["rotate"] == "true":
		movement_detect(args["inputstart"], args["mode"], True)
	else :
		movement_detect(args["inputstart"], args["mode"], False)
