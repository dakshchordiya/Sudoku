import cv2
import numpy as np
from tensorflow.keras.models import load_model



def initializePredictionModel():
    model = load_model("Image Classification/my_model.h5")
    return model




def preprocess(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg , (5,5) , 1)
    thresholdImg = cv2.adaptiveThreshold(blurImg, 255, 1, 1, 11, 2)
    return thresholdImg


def biggestContour(contours):
    biggest = np.array([])
    maxArea = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            peri = cv2.arcLength(contour , True)
            approx = cv2.approxPolyDP(contour, 0.02*peri, True)
            if area>maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest , maxArea



def reorder(points):
    points = points.reshape((4,2))
    newPoints = np.zeros((4,1,2) , dtype = np.int32)
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points , axis = 1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]
    return newPoints


    
def splitBoxes(img):
    rows = np.vsplit(img , 9)
    boxes = []
    for row in rows:
        columns = np.hsplit(row,9)
        for box in columns:
            boxes.append(box)
    return boxes



def getPrediction(boxes , model):
    results = []
    for image in boxes:
        img = np.asarray(image)
        img = img[4:img.shape[0]-4 , 4:img.shape[1]-4]
        img = cv2.resize(img , (32,32))
        img = img/255.
        img = img.reshape(1,32,32,1)
        
        predictions = model.predict(img)
        classIndex = np.argmax(predictions , axis = 1)
        probability = np.amax(predictions)
        #print(classIndex , probability)
        if probability > 0.8:
            results.append(classIndex[0])
        else:
            results.append(0)

    return results


  
def displayNumbers(image , numbers , color = (0,255,0)):
    secH = int(image.shape[0] / 9)
    secW = int(image.shape[1] / 9)
    for x in range(9):
        for y in range(9):
            if numbers[9*y +x] != 0:
                cv2.putText(image , str(numbers[9*y + x]) , (x*secW + int(secW/2) - 10 , int((y+0.8)*secH)) , cv2.FONT_HERSHEY_COMPLEX ,
                                  1 , color , 2 , cv2.LINE_AA )
    return image            




def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img



#### 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale):
    rows = len(imgArray)
    columns = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range (rows):
            for y in range(columns):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        blankImg = np.zeros((height, width, 3), np.uint8)
        hor = [blankImg]*rows
        hor_con = [blankImg]*rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver



##########################################################################################################################################


