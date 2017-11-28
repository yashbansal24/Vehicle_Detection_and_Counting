"""

Author: Yash Bansal / Rahul Ladda

Vehicle Detection and counting using Image Processing

"""

"""
The main idea is to capture a background Subtractor
using models of gaussian. Later the noise is removed
by thresholding and morphological operations.

"""


import numpy as np
import cv2

coords=[[5,51],[6,372],[485,58]]
#coords=[[5,51][][][],[6,372][][][],[485,58][][][]]

threshold = 600
"""
Helper function for calculating euclidean distance.
"""
def distance(x, y, type='euclidian', x_weight=1.0, y_weight=1.0):
    if type == 'euclidian':
        return math.sqrt(float((x[0] - y[0])**2) / x_weight + float((x[1] - y[1])**2) / y_weight)

"""
This function is calculating the centroid.
This takes two arguments x,y (old centroid)
 and new points (w,h).
 Then take the avg of all these.
"""

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)

"""
Opencv api reference:
Labels is a matrix the size of the input image where each element has a value equal to its label.

Stats is a matrix of the stats that the function calculates. It has a length equal to the number of labels and a width equal to the number of stats. It can be used with the OpenCV documentation for it:

Statistics output for each label, including the background label, see below for available statistics. Statistics are accessed via stats[label, COLUMN] where available columns are defined below.

cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
cv2.CC_STAT_WIDTH The horizontal size of the bounding box
cv2.CC_STAT_HEIGHT The vertical size of the bounding box
cv2.CC_STAT_AREA The total area (in pixels) of the connected component
Centroids is a matrix with the x and y locations of each centroid. The row in this matrix corresponds to the label number.


"""
def cluster_analysis(frame,i):
    src = frame
    # You need to choose 4 or 8 for connectivity type
    connectivity = 8
    # Perform the operation
    output = cv2.connectedComponentsWithStats(frame, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    #print num_labels
    # The second cell is the label matrix
    labels = output[1]

    #print np.count_nonzero(labels)
    # The third cell is the stat matrix
    stats = output[2]
    #print stats
    # The fourth cell is the centroid matrix
    centroids = output[3]
    #print centroids
    k=0
    lis=[]
    for l in stats:
        if l[4]>=threshold:
            lis.append([l[0],l[1],l[0]+l[2], l[1]+l[3]])
    k+=1
    return lis



"""
Capturing the video. opencv provides an
inbuilt api function to work for videos.
It extracts the image one by one.

"""
cap = cv2.VideoCapture('input.mp4')
ret, frame = cap.read()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))

np.zeros
pts = np.asarray(coords, np.int32)
#pts = np.asarray(coords, np.int32)
#pts = pts.reshape((-1,1,2))
#cv2.polylines(img,[pts],True,(0,255,255))
#mask = np.full((frame.shape),255)
#kernel = np.ones((3,3),np.uint8)
fgbg = cv2.createBackgroundSubtractorMOG2()
i=0

"""
This loop will run till the end of the video.
For chaging threshold, change the values at the
declarations at the top.

"""
while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

######## Apply Morphological Filtering

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2)


######### Apply thresholding. There is no need
######### for Ostu Method. We can remove the shadows from the obtained frame.

    ### shadows turns to backgrounds
    #ret2,img = cv2.threshold(fgmask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, img = cv2.threshold(fgmask, 127, 255, 0)

    #####  Applying Cluster Analysis



    lis = cluster_analysis(img,i)
    for l in lis:
        frame = cv2.rectangle(frame,(l[0],l[1]),(l[2],l[3]),(0,255,0),3)

    #######  Drawing the text on the top corner of video frame.
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,str(len(lis)-1),(60,60), font, 2,(255,255,255))
    #frame = cv2.fillPoly( frame, pts, 0 )

    ### Printing the frame.
    cv2.imshow('frame',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    i+=1

#
cap.release()
cv2.destroyAllWindows()
