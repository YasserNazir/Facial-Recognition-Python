from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import copy
from datetime import datetime

dateTimeObj = datetime.now()

# python3 recognize_faces_video.py --encodings encoding/encodings.pickle --output output/webcam_face_recognition_output.avi --display 1

#################SETTINGS#####################
widthpx = 400
time.sleep(2.0)
framecount = 0
framecountlimit = 10000
frameinit = 5
frametime = frameinit + 10
framecounter = 0
repeater = 1
executionend = 0
#############################################

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up



print("[INFO] starting video stream...")

stream = ('nvcamerasrc ! '
                       'video/x-raw(memory:NVMM), '
                       'width=(int)1280, height=(int)720, '
                       'format=(string)I420, framerate=(fraction)24/1 !'
                       'nvvidconv ! '
                       'video/x-raw, width=(int){}, height=(int){}, '
                       'format=(string)BGRx ! '
                       'videoconvert ! appsink').format(1280,720)

vs = VideoStream(src=1).start()
writer = None


namedict = {"Unknown": 0}

for i in range(len(data["names"])):
	temp={data["names"][i]:0}
	namedict.update(temp)

counter=0



confirmedstring = "No data "
	
confirmeddict=dict(namedict)

start = time.time()
# loop over frames from the video file stream
while framecount<framecountlimit:
	# grab the frame from the threaded video stream
	dim = (1280,720)
	frame = cv2.resize(vs.read(),dim,interpolation = cv2.INTER_AREA)

	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame,width=widthpx)
	r = frame.shape[1] / float(rgb.shape[1])
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	# loop over the facial embeddings
	data2 = copy.copy(data)
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data2["encodings"],
			encoding)
		name = "Unknown"
		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			nameindex = {}
			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				nameindex[name] = i
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
			delindex = nameindex.get(name)
			if delindex is True:
			    data2["names"][delindex]
			    data2["encodings"][delindex]

		# update the list of names
		if name not in names :
		    names.append(name)
		

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 0, 255), 2)

	for i in range(len(names)):
		val2 = str(names[i])
		val = int(namedict.get(val2))
		calculate = val + 1
		namedict[names[i]] = calculate
			
	namestring = "In picture :  "
	for i in names:
		namestring = namestring + i + ","
		
	cv2.putText(frame,namestring,(0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),2)
	framecount=framecount+1
	framecounter = framecounter + 1

	for i in range(len(names)):
		val2 = str(names[i])
		val = int(namedict.get(val2))
		calculate = val + 1
		namedict[names[i]] = calculate

	if framecounter == frametime :
		confirmedstring = "Confirmed presence within " + str(frametime) +"/"+ str(repeater) +" frames : "
		for k,v in namedict.items():
			if v >= frameinit :
				confirmedstring += str(k)
				confirmedstring += ", "
				confirmeddict[k]+= 1 
			namedict[k] = 0
		framecounter = 0
		repeater = repeater + 1
		
	cv2.putText(frame,confirmedstring,(0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
	
	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)
	# if the writer is not None, write the frame with recognized
	# faces to disk
	if writer is not None:
		writer.write(frame)

	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break


	
	


	
	
end = time.time()
totaltime = end - start
fps = framecount/totaltime
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

text = "logs/"+str(dateTimeObj)+".txt"
f= open(text,"w+")
f.write("\n*********************************************************\n")
print("*********************************************************")
f.write("LOG : %s " % str(dateTimeObj))
print("LOG : ",dateTimeObj)
f.write("\n*********************************************************\n")
print("*********************************************************")
f.write("Total Frame Count : %d \n" % framecount)
print("Total Frame Count : ",framecount)
f.write("FPS : %d \n" % fps)
print("FPS : ",fps)
f.write("Execution time (s) : %s \n" % str(totaltime))
print("Execution time (s) : ",totaltime)
f.write("Frame Recognize Limit : %s \n" % str(frameinit))
print("Frame Recognize Limit : ",frameinit)

print("\nResults : \n",totaltime)
f.write("\nResults : %s \n" % str(totaltime))
for k,v in confirmeddict.items():
	stringtemp = str(k)+" = "+str(v)+"/"+str(repeater)+"("+str((v/repeater)*100)+" %)\n"
	print(stringtemp)
	f.write(stringtemp)
f.close()


