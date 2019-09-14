# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import threading
from detectors import yolo_detect, color_detect, movement_detect

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", required=True, help="Start or end gate")
ap.add_argument("-i", "--input", required=True, help="Select input")
ap.add_argument("-t", "--type", required=True, help="Set detection type")
ap.add_argument("-r", "--rotate", required=False, help="Rotate image")
args = vars(ap.parse_args())

if args["type"] == "yolo":
	yolo_detect(args["input"], args["mode"])
if args["type"] == "color":
	color_detect(args["input"], args["mode"])
if args["type"] == "movement":
	if args["rotate"] == "true":
		movement_detect(args["input"], args["mode"], True)
	else :
		movement_detect(args["input"], args["mode"], False)
