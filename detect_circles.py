#import the necessary packages
import numpy as np
import argparse
import cv2
import time

cap = cv2.VideoCapture(0) # Set Capture Device, in case of a USB Webcam try 1, or give -1 to get a list of available devices

#Set Width and Height
# cap.set(3,1280)
# cap.set(4,720)

# The above step is to set the Resolution of the Video. The default is 640x480.
# This example works with a Resolution of 640x480.
boundaries = [
        #([17, 15, 40], [70, 66, 200]), # R >= 100, B >= 15, G >= 17 and R <= 200, B <= 56, G <= 50
        ([40, 31, 4], [220, 88, 80])
]

def findCircles(image) :
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
	gray = cv2.GaussianBlur(gray,(5,5),0);
	gray = cv2.medianBlur(gray,5)

	# Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
	gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3.5)
    	kernel = np.ones((3,3),np.uint8)


	#kernel = np.ones((3,3)),np.uint8)
    	gray = cv2.erode(gray,kernel,iterations = 1)
	# gray = erosion

	gray = cv2.dilate(gray,kernel,iterations = 1)
	# gray = dilation

	# get the size of the final image
	# img_size = gray.shape
	# print img_size

	# detect circles in the image
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=45, minRadius=0, maxRadius=0)
	return circles

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# load the image, clone it for output, and then convert it to grayscale

	output = frame.copy()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
	gray = cv2.GaussianBlur(gray,(5,5),0);
	gray = cv2.medianBlur(gray,5)

	# Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
	gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3.5)
    	kernel = np.ones((3,3),np.uint8)


	#kernel = np.ones((3,3)),np.uint8)
    	gray = cv2.erode(gray,kernel,iterations = 1)
	# gray = erosion

	gray = cv2.dilate(gray,kernel,iterations = 1)
	# gray = dilation

	# get the size of the final image
	# img_size = gray.shape
	# print img_size

	# detect circles in the image
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=45, minRadius=0, maxRadius=0)
	# print circles

	# ensure at least some circles were found
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")

		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle in the image
			# corresponding to the center of the circle
			#If circle radius is over 160, skip and don't draw it on image
			if(r > 160) :
				continue
			# Check if colors of found circle are blue and red

			cv2.circle(output, (x, y), r, (0, 255, 0), 4)
			#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
			#cv2.rectangle(output, (x - (r + 10), y - (r + 10)), (x + (r + 10), y + (r + 10)), (0, 128, 255), -1)
			roi = frame[y - (r + 10):y + (r + 10), x - (r + 10): x + (r + 10)]
			
			for (lower, upper) in boundaries:
				# Create NumPy arrays from the boundaries
        			lower = np.array(lower, dtype = "uint8")
        			upper = np.array(upper, dtype = "uint8")
       				# Find the colors within the specified boundaries and apply the mask
				output2 = roi.copy()
        			mask_1 = cv2.inRange(roi, lower, upper)
				res = cv2.bitwise_and(roi,roi, mask=mask_1)
				cv2.imshow('boundaries',res)
				cv2.waitKey(100)

				detected = findCircles(res)
				#print(detected)
				if detected is not None:
					# convert the (x, y) coordinates and radius of the circles to integers
					detected = np.round(detected[0, :]).astype("int")

					# loop over the (x, y) coordinates and radius of the circles
					for (x, y, r) in detected:
					# draw the circle in the output image, then draw a rectangle in the image
					# corresponding to the center of the circle
					#If circle radius is over 160, skip and don't draw it on image
						if(r > 160) :
							continue
							# Check if colors of found circle are blue and red

						cv2.circle(output2, (x, y), r, (0, 255, 0), 4)
						cv2.imshow('color 2',output2)
						cv2.waitKey(100)

	# Display the resulting frame
	#cv2.imshow('roi',roi)
	#cv2.imshow('gray',gray)
    	cv2.imshow('frame',output)
 	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
