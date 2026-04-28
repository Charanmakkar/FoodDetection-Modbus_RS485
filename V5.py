from yolov5 import YOLOv5
import cv2
from time import sleep

# set model params
model_path = "last.pt"
device = "cpu" #"cuda:0" # or "cpu"

# init yolov5 model
yolov5 = YOLOv5(model_path, device)

# Setting up WEB cam as camera to capture image
cam = cv2.VideoCapture(1)

def click_image():
	try:
		ret, img = cam.read()		
		# img = cv2.imread("o2.jpg")
		img_name = "orignal.jpg"
		cv2.imwrite(img_name, img)

##                Co-ordiinates updated on 09122022 10:10AM
##                img = cv2.rectangle(img, (0, 190), (320, 480), color, thickness)
##                img = cv2.rectangle(img, (0, 0), (310, 200), color, thickness)
##                img = cv2.rectangle(img, (350, 0), (640, 220), color, thickness)
##                img = cv2.rectangle(img, (310, 200), (640, 480), color, thickness)

		cropped_image1 = img[190:480, 0:320] # Slicing to crop the image       R1
		img_name = "R1.jpg"
		cv2.imwrite(img_name, cropped_image1)

		cropped_image2 = img[0:200, 0:310] # Slicing to crop the image        R2
		img_name = "R2.jpg"
		cv2.imwrite(img_name, cropped_image2)

		cropped_image3 = img[0:220, 350:640] # Slicing to crop the image      R3
		img_name = "R3.jpg"
		cv2.imwrite(img_name, cropped_image3)

		cropped_image4 = img[200:480, 310:640] # Slicing to crop the imag     R4
		img_name = "R4.jpg"
		cv2.imwrite(img_name, cropped_image4)

		print("Image click successful")
	except:
		print("ERROR IN CLICKING IMAGAE")
		print("Please check camera is attached or Restart me once")


def objdetection(image_path = "original.jpg"):
	image1 = image_path

	results = yolov5.predict(image1, size=720)#, augment=True)
	# print(results)

	# parse results
	predictions = results.pred[0]
	boxes = predictions[:, :4] # x1, y1, x2, y2
	scores = predictions[:, 4]
	categories = predictions[:, 5]

	# show detection bounding boxes on image
	results.show()

	c = 0
	list_of_results = []

	for x in results.crop():
	    value = (x['label'])
	    if "car" in value :
	        c = c+1
	        list_of_results.append(value.split(" ")[0])
	        print(value)
	    if "truck" in value :
	        c = c+1
	        list_of_results.append(value.split(" ")[0])
	        print(value)
	    if "bus" in value :
	        c = c+1
	        list_of_results.append(value.split(" ")[0])
	        print(value)
	    if "cell" in value :
	        c = c+1
	        list_of_results.append(value.split(" ")[0])
	        print(value)

	print("total object detected = ", c)
	# print(list_of_results)
	finalCount = len(list_of_results)
	print("finalCount = ", finalCount)
	return c, list_of_results


# objdetection("R1.jpg")
# objdetection("R2.jpg")
# objdetection("R3.jpg")
# objdetection("R4.jpg")

while(1):
    click_image()

    #this is the function that runs finds objects present in the image
    try:
        resultRoad1, a = objdetection("R1.jpg")
##        resultRoad2, b = objdetection("R2.jpg")
        resultRoad4, c = objdetection("R3.jpg")
##        resultRoad3, d = objdetection("R4.jpg")
        
    except:
        print("Error with passing cropped files")

    try:
##        writeToJSON(resultRoad1*109, resultRoad2*109, resultRoad3*109, resultRoad4*109)
        readJSON()
    except:
        print("Error with passing value(s) for JSON file")
    
    # sleep(sleep_delay_in_loop)
