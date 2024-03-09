"""
OpenCV is the largest open source computer vision and machine learning software library. The library contains more
than 2500 optimized algorithms, used extensively in the field of  computer vision and machine learning algorithms. 
"""

""" Importing OpenCV """
import cv2 as cv
import numpy as np

""" Reading and Displaying Image """
def displayImage(txt, img):
    cv.imshow(txt, img)                                       # Displaying image
    cv.waitKey(0)                                             # Keeping image open [ 0 : infinite time ]
# img = cv.imread("OpenCV/Data/Images/4.jpg")                      # Reading image
# displayImage('Image', img)

""" Reading and Displaying Video """
def displayVid(vid):
    while True:
        isTrue, frame = vid.read()                            # Reading frames [ one at a time ]
        if frame is None:                                     # Checking if video has not ended
            break
        cv.imshow('Video', frame)                             # Displaying the frame
        # cv.imshow('ResizedVideo', rescale(frame, 0.5))      # Displaying the resized frame

        if cv.waitKey(33) and 0xFF==ord('q'):                 # Waiting  and checking for key triggers
            break

    vid.release()                                             # Release resources associated with the capture
    cv.destroyAllWindows()                                    # Close all HighGUI windows
# vid = cv.VideoCapture("OpenCV/Data/Videos/1.mp4")                # Reading video [ 0 for webcam ]
# displayVid(vid)

""" Drawing & Writing on Images """
# Creating a blank image
def createBlank(height, width, channels=1):
    return np.zeros((height, width, channels), dtype='uint8')
# cv.imshow('Blank', createBlank(500, 500, 3))

# Changing color [ B, G, R ]
# blank[200:300, 200:300] = 238, 153, 102                                 # blank[:] for complete colouring
# cv.imshow('Minet', blank)

# Drawing a Line
# cv.line(blank, (250, 250), (250, 400), (255, 255, 255), thickness=3)
# cv.imshow('Line', blank)

# Drawing a Rectangle / Square
# cv.rectangle(blank, (100, 100), (400, 400), (0, 0, 255), thickness=3)   # set thickness = cv.FILLED/-1 for solid color
# cv.imshow('Rectangle', blank)

# Drawing a Circle
# cv.circle(blank, (250, 250), 150, (0, 255, 0), thickness=2)             # set thickness = cv.FILLED/-1 for solid color
# cv.imshow('Circle', blank)

# Writing Text
# cv.putText(blank, 'Sup Loser!', (161, 175), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 5)
# cv.imshow('Text', blank)

# cv.waitKey(0)
# displayImage('Final', blank)

""" Basic Functions [ Resize / Crop / Translate / Rotate / Flip ] """
# img = cv.imread("OpenCV/Data/Images/3.jpg")                      # Reading image
# cv.imshow('Original', img)

# Rescaling an Image
def rescale(frame, scale=1):                                  # Usable for images, videos and live video
    height = int(frame.shape[0] * scale)                      # Scaling height
    width = int(frame.shape[1] * scale)                       # Scaling width
    dim = (width, height)                                     # Creating new dimension array
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA) # Returning the resized frame
# img = rescale(img, 0.5)
# cv.imshow('Rescaled', img)

# Resizing an Image [ .INTER_AREA : scaling down ; .INTER_LINEAR (faster) / .INTER_CUBIC (better quality) : enlarging ]
# resized = cv.resize(img, (400, 300), interpolation=cv.INTER_AREA)
# cv.imshow('Resized', resized)

# Cropping an Image
def crop(img, x1, x2, y1, y2):
    return img[y1:y2, x1:x2]
# cv.imshow('Cropped', crop(img,  250, 375, 125, 300))

# Translating an Image
def translate(img, x, y):                                                   # +ve x : right , +ve y : down
    import numpy as np
    transMat = np.float32([[1, 0, x], [0, 1, y]])                           # Creating a translation matrix
    dimensions = (img.shape[1], img.shape[0])                               # Getting the dimensions
    return cv.warpAffine(img, transMat, dimensions)                         # Applying the translation matrix
# cv.imshow('Translated', translate(img, 50, 50))

# Rotating an Image
def rotate(img, angle, rotPoint=None):                                      # -ve angle : clockwise
    height, width = img.shape[:2]                                           # Getting the dimensions
    if rotPoint is None:                                                    # Setting the rotation point
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1)                     # Creating a rotation matrix
    dimensions = (width, height)                                            # Getting the dimensions
    return cv.warpAffine(img, rotMat, dimensions)                           # Applying the rotation matrix
# cv.imshow('Rotated', rotate(img, -10, (0, 0)))
# cv.imshow('RotatedRotate', rotate(rotate(img, -10, (0, 0)), 10, (0, 0)))  # Data is lost in rotation
# for i in range(361):
#     cv.imshow('Rotating', rotate(img, i))
#     if cv.waitKey(5) and 0xFF==ord('q'):
#         break

# Flipping an Image
def flip(img, flipCode):                                                    # 0 : vertical , 1 : horizontal , -1 : both
    return cv.flip(img, flipCode)
# cv.imshow('Flipped', flip(img, -1))

# cv.waitKey(0)

""" Intermediate Functions [ Gray / Blur / Edge-Detect / Dilate / Erode ] """
# img = rescale(cv.imread("OpenCV/Data/Images/3.jpg"), 0.5)
# cv.imshow('Original', img)

# Converting to Grayscale
def gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Grayscale', gray(img))

# Blurring an Image
def blur(img):
    return cv.GaussianBlur(img, (5, 5), 0)
# cv.imshow('Blurred', blur(img))

# Edge Detection
def canny(img, minVal, maxVal):
    return cv.Canny(img, minVal, maxVal)
# cv.imshow('Canny', canny(blur(img), 125, 175))                                # Blur image to reduce edge count

# Image Dilation
def dilate(img, kernel, iterations):
    return cv.dilate(img, kernel, iterations=iterations)
# cv.imshow('Dilated', dilate(canny(blur(img), 125, 175) , (3, 3), 1))

# Image Erosion
def erode(img, kernel, iterations):                                             # Reverse of dilation, in some sense
    return cv.erode(img, kernel, iterations=iterations)
# cv.imshow('Eroded', erode(dilate(canny(blur(img), 125, 175) , (3, 3), 1), (3, 3), 1))

# cv.waitKey(0)

""" Contours Detection """
# img = cv.imread("OpenCV/Data/Images/3.jpg")                                          # Reading image
# cv.imshow('Original', img)

# pre = canny(blur(gray(rescale(img, 0.5))), 125, 175)                            # Preprocessing the image
# _, pre = cv.threshold(gray(rescale(img, 0.5)), 125, 255, cv.THRESH_BINARY)      # Binarise (B&W) the image
# cv.imshow('Preprocessed', pre)

# contours, hierarchy = cv.findContours(pre, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)# Finding contours
# print(f'{len(contours)} contours found!')

# blank = createBlank(img.shape[0]//2, img.shape[1]//2, 3)                        # Creating a blank image
# cv.drawContours(blank, contours, -1, (0, 0, 255), 1)                            # Drawing contours
# cv.imshow('Contours', blank)

# cv.waitKey(0)                                                                   # Keeping the image open
