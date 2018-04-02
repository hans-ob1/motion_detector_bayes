import numpy as np
import cv2

# Pre-setup of frame capture
frame_width = 1280
frame_height = 720
frame_fps = 30

cap = cv2.VideoCapture('car_thief.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, frame_fps)

bgsubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

# initalize params
firstObervation = True
isMovingProbability = 0

frame_counter = 0

def calculateMovement(imgGrayScale, thres):
	height, width = imgGrayScale.shape
	areaTotal = height*width
	activePixelsCount = cv2.countNonZero(imgGrayScale)
	motionPercentage = (activePixelsCount / areaTotal)*100

	if motionPercentage > thres:
		return True
	else:
		return False


def bayesianClassifier(isMoving):

	global isMovingProbability
	global firstObervation

	if firstObervation:
		if isMoving:
			isMovingProbability = (0.8 * 0.5) / 0.9
		else:
			isMovingProbability = (0.1 * 0.5) / 0.9

		firstObervation = False

	else:
		if isMoving:
			isMovingProbability = (0.8 * isMovingProbability) / ((0.8 * isMovingProbability) + 0.1 * (1 - isMovingProbability))
		else:
			isMovingProbability = (0.1 * isMovingProbability) / ((0.1 * isMovingProbability) + 0.8 * (1 - isMovingProbability))

	# boundary setting to avoid negatives:
	if isMovingProbability > 0.6:
		isMovingProbability = 0.6

	if isMovingProbability < 0.1:
		isMovingProbability = 0.1

	return isMovingProbability > 0.5	#return true if probability more than 0.5




def main():

	global frame_counter

	while True:
	    ret, frame = cap.read()
	    frame_bg = frame.copy()
	    
	    # apply filters
	    step1 = cv2.medianBlur(frame_bg,5)
	    step2 = bgsubtractor.apply(step1)
	    step3 = cv2.GaussianBlur(step2,(15,15),0)
	    _,bgMaskProcessed= cv2.threshold(step3,10,255,cv2.THRESH_BINARY)

	    isMoving = calculateMovement(bgMaskProcessed, 0.3)

	    if (bayesianClassifier(isMoving)):
	    	indicatorString = "Motion Detected!"
	    	indicatorTagColour = (0,255,0)
	    else:
	    	indicatorString = "No Motion"
	    	indicatorTagColour = (0,0,255)

	    # draw indication on frame
	    indicatorTagLocation = (30,30)
	    cv2.putText(frame, 
	    			indicatorString, 
	    			indicatorTagLocation, 
	    			cv2.FONT_HERSHEY_SIMPLEX, 
	    			1, 
	    			indicatorTagColour, 
	    			2, 
	    			cv2.LINE_AA)

	    if frame_counter > 169 and frame_counter < 226:

		    filename = str(frame_counter).zfill(4)
		    filename += ".png"
		    print("Writing: " + filename)

		    frame_to_copy = frame.copy()
		    f_height, f_width = frame_to_copy.shape[:2]
		    resized = cv2.resize(frame_to_copy,(640,360), interpolation = cv2.INTER_CUBIC)

		    cv2.imwrite(filename, resized)

	    frame_counter += 1

	    # display frame
	    cv2.imshow('frame',frame)

	    k = cv2.waitKey(30) & 0xff
	    if k == 27:
	    	break
	    

	cap.release()
	cv2.destroyAllWindows()


main()