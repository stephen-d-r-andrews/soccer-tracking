# USAGE
# python tracking.py --input videos/inputvideo.mp4 --output output/outputvideo.avi --anchors videos/inputvideo.json --yolo yolo-coco

# code based on https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/   (Adrian Rosebrock)

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import json
import copy

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-o", "--output", required=True,
                help="path to output video")
ap.add_argument("-a", "--anchors", required=True,
                help="path to anchor file")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

def find_intersection(a):
        if a[2]!=a[0] and a[4]!=a[6]:
                m0 = (a[3]-a[1])/(a[2]-a[0])
                m1 = (a[5]-a[7])/(a[4]-a[6])
                c0 = a[1]-m0*a[0]
                c1 = a[5]-m1*a[4]
                x = (c1-c0)/(m0-m1)
                y = m0*x+c0
        elif a[2]==a[0]:
                m1 = (a[5]-a[7])/(a[4]-a[6])
                c1 = a[5]-m1*a[4]
                x = a[0]
                y = m1*x+c1
        else: # a[4]==a[6]:
                m0 = (a[3]-a[1])/(a[2]-a[0])
                c0 = a[1]-m0*a[0]
                x = a[4]
                y = m0*x+c0
        return (x,y)
    
def find_real_point(x,y,a,realx0,realx1,realy0,realy1):
        [vx, vy] = find_intersection(a)
        [wx, wy] = find_intersection([a[0],a[1],a[6],a[7],a[2],a[3],a[4],a[5]]) 

        [gx, gy] = find_intersection([a[6],a[7],a[4],a[5],wx,wy,x,y])                
        [hx, hy] = find_intersection([a[2],a[3],a[4],a[5],vx,vy,x,y])

        #print(vx,vy,wx,wy,gx,gy,hx,hy)

        A = (realy1-realy0)/((1./((a[7]-vy)/(a[5]-vy)))-1.)
        B = realy0-A
        Y = A/((gy-vy)/(a[5]-vy))+B
        C = (realx0-realx1)/((1./((a[2]-wx)/(a[4]-wx)))-1.)
        D = realx1-C
        X = C/((hx-wx)/(a[4]-wx))+D

        #print(A,B,C,D)
        return X,Y


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# read in the anchor data
if os.path.exists(args["anchors"]):
        anchorDict = json.load(open(args["anchors"],"r"))
             
else:
        print("Anchor file does not exist")
        anchorDict = {}


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

# loop over frames from the video file stream
#while True:

lk_params = dict(winSize=(15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def mark_image(event,x,y,flags,param):
        # print(x,y)
        global ganchors
        global gptr
        global ganchor_change 
        if event == cv2.EVENT_LBUTTONDOWN:
                if state == ord('s'):
                        print("physical coords for screen coords ("+str(x)+","+str(y)+")")
                        phys_str = input().strip().split()
                        ganchors.append([x,y,float(phys_str[0]),float(phys_str[1])])
                elif state == ord('m'):
                        ganchors[gptr][0] = x
                        ganchors[gptr][1] = y
                        gptr += 1

        '''
        global lclick, rclick, gx, gy
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE:
                gx = x
                gy = y
        if event == cv2.EVENT_LBUTTONDOWN:
                if stickyk in [ord('a'), ord('p')]:
                        lclick = True
        if event == cv2.EVENT_RBUTTONDOWN:
                if stickyk in [ord('a'), ord('p')]:
                        rclick = True
        '''

ganchors = []
#ganchors = [[436, 369, 60.0, 73.33], [710, 326, 60.0, 20.0], [987, 348, 90.0, 20.0], [754, 400, 90.0, 73.33]]
gptr = 0
state = ord('s')
#cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image',mark_image)
framerate = 30
        
framedist = 0.
framecnt = 0
oldplayerlist = []
annotationStore = []

for framenum in range(10000):
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
                break

        if framenum<-20:
                continue
        
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
                (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        if str(framenum) in anchorDict.keys():
                ganchors = copy.deepcopy(anchorDict[str(framenum)])

        for i in range(len(ganchors)):
                frame = cv2.circle(frame,(ganchors[i][0],ganchors[i][1]),5,(255,0,0),-1)
        ganchor_change = False
        anchor_period = framerate//5
        
        if framenum%anchor_period==0:
                if str(framenum) not in anchorDict.keys():
                        cv2.imshow('image',frame)
                        while True:
                                k = cv2.waitKey(10) & 0xFF
                                if k == ord('d'):
                                        break
                                elif k == ord('s'):
                                        ganchors = []
                                        state = k
                                elif k == ord('m'):
                                        gptr = 0
                                        state = k
                        anchorDict[str(framenum)] = copy.deepcopy(ganchors)
                print(framenum,ganchors)
                json.dump(anchorDict,open(args["anchors"],"w"))
                        
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
                        if confidence > args["confidence"] and LABELS[classID]=="person":
                                # scale the bounding box coordinates back relative to
                                # the size of the image, keeping in mind that YOLO
                                # actually returns the center (x, y)-coordinates of
                                # the bounding box followed by the boxes' width and
                                # height
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top
                                # and and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates,
                                # confidences, and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes,confidences,args["confidence"],args["threshold"])
        optflow_period = 3*framerate
        if framenum%optflow_period==0:
                p2 = []
        # ensure at least one detection exists

        cv2.putText(frame,"framedist {:.0f}".format(framedist)+" "+str(framecnt),(50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        playerlist = []
        if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        #text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                        
                        if len(ganchors)==4:
                                anchorvec = [ganchors[0][0],ganchors[0][1],ganchors[1][0],ganchors[1][1],ganchors[2][0],ganchors[2][1],ganchors[3][0],ganchors[3][1]]
                                realx0 = ganchors[1][2]
                                realy0 = ganchors[1][3]
                                realx1 = ganchors[3][2]
                                realy1 = ganchors[3][3]
                                X,Y = find_real_point(int(x+w/2),int(y+h),anchorvec,realx0,realx1,realy0,realy1)
                                #text = text + "{:.2f} {:.2f}".format(X,Y)
                                playerlist.append([x,y,X,Y,w,h,0,[(X,Y)]])

                        if framenum%optflow_period == 0:
                                p2.append(np.array([1.*int(x+w/2),1.*int(y+h/2)]))

        if framenum%anchor_period==0:
                annotationStore = []
                textoverlay = np.zeros_like(frame)
                
                for p in playerlist:
                        [x,y,X,Y,w,h,playerid,pointlist] = p
                        if len(oldplayerlist)>0:
                                match_p = np.argmin(np.array([np.linalg.norm((X-pp[2],Y-pp[3])) for pp in oldplayerlist]))
                                speed = np.linalg.norm((X-oldplayerlist[match_p][2],Y-oldplayerlist[match_p][3]))
                                if speed<10:
                                        p[6] = oldplayerlist[match_p][6]
                                        if len(oldplayerlist[match_p][7])>0:
                                                p[7] = p[7]+oldplayerlist[match_p][7][:min(len(oldplayerlist[match_p]),5)]
                                        longspeed = np.linalg.norm((p[7][0][0]-p[7][-1][0],p[7][0][1]-p[7][-1][1]))*(5/(len(p[7])-1))
                                        if len(p[7])>=6:
                                                text = "{:.2f}".format(longspeed)
                                                red = min(255,int(255*(longspeed/30)))
                                                blue = 255-red                                                  
                                                #cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, (blue,0,red), 4)
                                                annotationStore.append(['t', text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, (blue,0,red), 4])
                                                #cv2.rectangle(textoverlay, (x, y), (x + w, y + h), (blue,0,red), 2)
                                                annotationStore.append(['c',(W-360+int(X),int(Y)),5,(blue,0,red),-1])
                                                #cv2.circle(frame,(W-360+int(X),int(Y)),5,(blue,0,red),-1)
                                        else:
                                                text = "{:d}".format(p[6])
                                                #cv2.putText(textoverlay, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 4)
                                                annotationStore.append(['c',(W-360+int(X),int(Y)),5,(0,0,0),1])
                                                #cv2.circle(frame,(W-360+int(X),int(Y)),5,(0,0,0),1)
                                else:
                                        p[6] = np.random.randint(5000)
                                        text = "{:d}".format(p[6])
                                        annotationStore.append(['c',(W-360+int(X),int(Y)),5,(0,0,0),1])
                                        #cv2.circle(frame,(W-360+int(X),int(Y)),5,(0,0,0),1)
                                        #cv2.putText(textoverlay, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 4)
                                        #cv2.rectangle(textoverlay, (x, y), (x + w, y + h), (0,255,255), 2)

                oldplayerlist = copy.deepcopy(playerlist)

        cv2.rectangle(frame, (W-360, 0), (W, 200), (255,255,255), -1)
        for a in annotationStore:
                if a[0] == 't':
                        cv2.putText(frame,a[1],a[2],a[3],a[4],a[5],a[6])
                elif a[0] == 'c':
                        cv2.circle(frame,a[1],a[2],a[3],a[4])
                        
        #frame = cv2.addWeighted(frame,1.,textoverlay,1.,0)
        cv2.imshow('image',frame)
        k = cv2.waitKey(1000//framerate) & 0xFF
        #cv2.imshow('image',textoverlay)
        #k = cv2.waitKey(1000//framerate) & 0xFF
                
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if framenum%optflow_period == 0:
                old_gray = frame_gray.copy()                
                old_frame = frame.copy()                
                feature_params = dict( maxCorners = 100,
                                       qualityLevel = 0.3,
                                       minDistance = 7,
                                       blockSize = 7 )        
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
                for i in range(len(p2)):
                        p0[i,0,0] = p2[i][0]
                        p0[i,0,1] = p2[i][1]
                p0 = np.copy(p0[:len(p2),:,:])
                mask = np.zeros_like(frame)
                        
        framedist = np.linalg.norm(frame.astype(float)-old_frame.astype(float))
        framecnt = np.count_nonzero(frame-old_frame)
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1 #[st==1]
        good_old = p0 #[st==1]

        # draw the tracks
        '''
        for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                color = [int(cc) for cc in COLORS[i%80]]
                if i<len(p0):
                        mask = cv2.line(mask, (a,b),(c,d), color, 2)
                        frame = cv2.circle(frame,(a,b),5,color,-1)
                        frame = frame+mask
        '''
        
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        old_frame = frame.copy()
        p0 = good_new.reshape(-1,1,2)
                                
        # check if the video writer is None
        if writer is None:
                # initialize our video writer
#		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                         (frame.shape[1], frame.shape[0]), True)

                # some information on processing single frame
                if total > 0:
                        elap = (end - start)
                        print("[INFO] single frame took {:.4f} seconds".format(elap))
                        print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

        # write the output frame to disk
        writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
