import cv2 as cv
import numpy as np
from skimage import morphology, measure

SIZE =  37
titleWindow = "Window"
slider_max = 40
erodeKernel = np.ones((1, 1), np.uint8)
dilateKernel = np.ones((1, 1), np.uint8)
minSizeObject = 1
erodeKernel2 = np.ones((1, 1), np.uint8)
dilateKernel2 = np.ones((1, 1), np.uint8)
minSizeObject2 = 1

def updateImage():

    resultRidges = morphology.erosion(ridges, erodeKernel)
    resultRidges = morphology.dilation(resultRidges, dilateKernel)
    resultRidges = morphology.remove_small_objects(resultRidges.astype(bool), min_size=minSizeObject)
    #resultRidges = morphology.medial_axis(resultRidges)
    resultRidges = resultRidges.astype(np.uint8)*255

    resultValleys = morphology.erosion(valleys, erodeKernel2)
    resultValleys = morphology.dilation(resultValleys, dilateKernel2)
    resultValleys = morphology.remove_small_objects(resultValleys.astype(bool), min_size=minSizeObject2)
    #resultValleys = morphology.medial_axis(resultValleys)

    resultValleys = resultValleys.astype(np.uint8)*127
    
    resultMap = resultRidges + resultValleys
    resultMap = cv.resize(resultMap, (1080, 720))


    resultMap = cv.cvtColor(resultMap, cv.COLOR_GRAY2BGR)
    
    resultMap[np.all(resultMap == [255,255,255], axis=-1)] = [0,0,255]
    resultMap[np.all(resultMap == [127,127,127], axis=-1)] = [255,0,0]
    
    cv.imshow(titleWindow, resultMap)
    return resultMap
    # print("UpdateImage")

def onErodeTrackbar(val):
    global erodeKernel
    erodeKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(val, val))
    #np.ones((val, val), np.uint8)
    updateImage()
def onDilateTrackbar(val):
    global dilateKernel
    dilateKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(val, val))
    updateImage()
def onRemoveSmallObjectsTrackbar(val):
    global minSizeObject
    minSizeObject = val
    updateImage()
def onErodeTrackbar2(val):
    global erodeKernel2
    erodeKernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(val, val))
    #np.ones((val, val), np.uint8)
    updateImage()
def onDilateTrackbar2(val):
    global dilateKernel2
    dilateKernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(val, val))
    updateImage()
def onRemoveSmallObjectsTrackbar2(val):
    global minSizeObject2
    minSizeObject2 = val
    updateImage()


valleys = np.load("Data/ValleyCross" + str(SIZE) + "x" + str(SIZE) + ".npy")
ridges = np.load("Data/RidgesCross" + str(SIZE) + "x" + str(SIZE) + ".npy")



cv.namedWindow(titleWindow)
erodeTrackbar_name = "Erosion"
cv.createTrackbar(erodeTrackbar_name, titleWindow , 1, slider_max, onErodeTrackbar)
dilateTrackbar_name = "Dilate"
cv.createTrackbar(dilateTrackbar_name, titleWindow , 1, slider_max, onDilateTrackbar)
removeSmallObjectsTrackbar_name = "Remove"
cv.createTrackbar(removeSmallObjectsTrackbar_name, titleWindow , 0, slider_max*100, onRemoveSmallObjectsTrackbar)
erodeTrackbar2_name = "Erosion2"
cv.createTrackbar(erodeTrackbar2_name, titleWindow , 1, slider_max, onErodeTrackbar2)
dilateTrackbar2_name = "Dilate2"
cv.createTrackbar(dilateTrackbar2_name, titleWindow , 1, slider_max, onDilateTrackbar2)
removeSmallObjectsTrackbar2_name = "Remove2"
cv.createTrackbar(removeSmallObjectsTrackbar2_name, titleWindow , 0, slider_max*100, onRemoveSmallObjectsTrackbar2)
# Show some stuff
onErodeTrackbar(1)
onDilateTrackbar(1)
onRemoveSmallObjectsTrackbar(1)
onErodeTrackbar2(1)
onDilateTrackbar2(1)
onRemoveSmallObjectsTrackbar2(1)
# Wait until user press some key
key = cv.waitKey()
if key == ord("s"):
    resultRidges = morphology.erosion(ridges, erodeKernel)
    resultRidges = morphology.dilation(resultRidges, dilateKernel)
    resultRidges = morphology.remove_small_objects(resultRidges.astype(bool), min_size=minSizeObject)

    resultValleys = morphology.erosion(valleys, erodeKernel2)
    resultValleys = morphology.dilation(resultValleys, dilateKernel2)
    resultValleys = morphology.remove_small_objects(resultValleys.astype(bool), min_size=minSizeObject2)

    np.save("Results/DetectedRidges.npy", resultRidges)
    np.save("Results/DetectedValleys.npy", resultValleys)
    
cv.destroyAllWindows()


