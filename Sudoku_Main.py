print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sudokuSolver

from utils import *

# =============================================================================

imagePath = 'Resources/1.jpg'
imgHeight = 450 
imgWidth = 450
model = initializePredictionModel()


# =============================================================================



### PREPARING THE IMAGE
img = cv2.imread(imagePath)
img = cv2.resize(img , (imgHeight ,imgWidth ))


blankImg = np.zeros((imgHeight , imgWidth , 3) , np.uint8)
thresholdImg = preprocess(img)


### FINDING THE CONTOURS
contourImg = img.copy()
biggestContourImg = img.copy()
contours , hierarchy = cv2.findContours(thresholdImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contourImg , contours , -1 , (0,255,0) , 3)

### FINDING THE BIGGEST CONTOUR AND USING IT AS SUDOKU
biggest , maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(biggestContourImg , biggest , -1 , (0,0,255) , 20)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0] , [imgWidth,0] , [0,imgHeight] , [imgWidth , imgHeight]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    warpColoredImg = cv2.warpPerspective(img, matrix, (imgWidth , imgHeight))
    detectedDigitsImg = blankImg.copy()
    warpGrayImg = cv2.cvtColor(warpColoredImg , cv2.COLOR_BGR2GRAY)


    ### SPLIT THE IMAGE AND FIND EACH DIGIT
    solvedDigitsImg = blankImg.copy()
    boxes = splitBoxes(warpGrayImg)
    # cv2.imshow("SAMPLE BOX" , boxes[1])
    numbers = getPrediction(boxes , model)
    # print(numbers)

    detectedDigitsImg = displayNumbers(detectedDigitsImg , numbers , color = (255,0,255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers>0 , 0 , 1)
    # print(posArray)
 
    
    ### FINDING SOLUTION OF SUDOKU
    board = np.array_split(numbers,9)
    # print(board)
    
    try:
        sudokuSolver.solve(board)
    except:
        pass
    # print(board)
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumber = flatList * posArray
    solvedDigitsImg = displayNumbers(solvedDigitsImg , solvedNumber )



    ### OVERLAY THE SOLUTION
    pts2 = np.float32(biggest)
    pts1 = np.float32([[0,0] , [imgWidth,0] , [0,imgHeight] , [imgWidth , imgHeight]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    warpColoredInvImg = img.copy()
    warpColoredInvImg = cv2.warpPerspective(solvedDigitsImg, matrix, (imgWidth , imgHeight))
    inv_perspective = cv2.addWeighted(warpColoredInvImg , 1 , img , 0.5 , 1)
    detectedDigitsImg = drawGrid(detectedDigitsImg)
    solvedDigitsImg = drawGrid(solvedDigitsImg)


    
    imgArray = ([img , contourImg , biggestContourImg , warpGrayImg] , 
                  [detectedDigitsImg , solvedDigitsImg , warpColoredInvImg , inv_perspective ])
    
    
    stackedImgs = stackImages(imgArray,0.8)
    
    cv2.imshow('Image Stack' , stackedImgs)
    cv2.waitKey(0)
    
    
else:
    print('No Sudoku Found!')
