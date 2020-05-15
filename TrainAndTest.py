import cv2
import numpy as np
import operator
import os

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


class ContourWithData():

    npaContour = None
    boundingRect = None
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    # calculate bounding rect
    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    # this is oversimplified, for a production grade program
    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA:
            return False
        return True


def main():
    allContoursWithData = []
    validContoursWithData = []

    try:
        # read in training classifications
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return

    try:
        # read in training images
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return

    # reshape numpy array to 1d, necessary to pass to call to train
    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))

    kNearest = cv2.ml.KNearest_create()

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    # read in testing numbers image
    #imgTestingNumbers = cv2.imread("test1.png")
    #imgTestingNumbers = cv2.imread("test2.png")
    #imgTestingNumbers = cv2.imread("test3.png")
    imgTestingNumbers = cv2.imread("imageTextN.png")
    #imgTestingNumbers = cv2.imread("training_chars.png")

    if imgTestingNumbers is None:
        print("error: image not read from file \n\n")
        os.system("pause")
        return

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)
    imgThresh = cv2.adaptiveThreshold(
        imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # make a copy of the thresh image, this in necessary because findContours modifies the image
    imgThreshCopy = imgThresh.copy()

    npaContours, npaHierarchy = cv2.findContours(
        imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for npaContour in npaContours:                             # for each contour
        # instantiate a contour with data object
        contourWithData = ContourWithData()
        # assign contour to contour with data
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(
            contourWithData.npaContour)
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight(
        )
        contourWithData.fltArea = cv2.contourArea(
            contourWithData.npaContour)
        # add contour with data object to list of all contours with data
        allContoursWithData.append(contourWithData)

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            # if so, append to valid contour list
            validContoursWithData.append(contourWithData)

    validContoursWithData.sort(key=operator.attrgetter(
        "intRectX"))         # sort contours from left to right

    # declare final string, this will have the final number sequence by the end of the program
    strFinalString = ""

    for contourWithData in validContoursWithData:            # for each contour
        # draw a green rect around the current char
        cv2.rectangle(imgTestingNumbers,
                      (contourWithData.intRectX, contourWithData.intRectY),
                      (contourWithData.intRectX + contourWithData.intRectWidth,
                       contourWithData.intRectY + contourWithData.intRectHeight),
                      (0, 255, 0),
                      2)

        imgROI = imgThresh[contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,
                           contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth]

        # resize image, this will be more consistent for recognition and storage
        imgROIResized = cv2.resize(
            imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        # flatten image into 1d numpy array
        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        # convert from 1d numpy array of ints to 1d numpy array of floats
        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(
            npaROIResized, k=1)

        strCurrentChar = str(chr(int(npaResults[0][0])))
        strFinalString = strFinalString + strCurrentChar
    print("\n" + strFinalString + "\n")                  # show the full string

    # show input image with green boxes drawn around found digits
    cv2.imshow("imgTestingNumbers", imgTestingNumbers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()             # remove windows from memory
    return


if __name__ == "__main__":
    main()
