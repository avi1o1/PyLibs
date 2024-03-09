import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import basics

""" Colour Channels """
# img = basics.rescale(cv.imread("OpenCV/Data/Images/3.jpg"), 0.5)
# cv.imshow('Original', img)

# Splitting the Image
# b, g, r = cv.split(img)
# cv.imshow('Blue', b)
# cv.imshow('Green', g)
# cv.imshow('Red', r)

# Visualizing the colour channels
# blank = basics.createBlank(img.shape[0], img.shape[1], 1)
# blue, red, green = cv.merge([b, blank, blank]), cv.merge([blank, g, blank]), cv.merge([blank, blank, r])
# cv.imshow('Blue', blue)
# cv.imshow('Green', green)
# cv.imshow('Red', red)

# Merging the Image
# merged = cv.merge([b, g, r])
# cv.imshow('Merged', merged)

# cv.waitKey(0)

""" Colour Spaces """
# img = basics.rescale(cv.imread("OpenCV/Data/Images/3.jpg"), 0.5)
# cv.imshow('Original', img)

# BGR to RGB
# rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# cv.imshow('RGB', rgb)

# BGR to Grayscale [ Gray = (B + G + R) / 3, thus colour information is lost ]
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Grayscale', gray)

# BGR <=> HSV [ H : Hue, S : Saturation, V : Value ]
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('HSV', hsv)
# bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
# cv.imshow('BGR', bgr)

# BGR <=> LAB [ L : Lightness, A : Green to Red, B : Blue to Yellow ]
# lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('LAB', lab)
# bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
# cv.imshow('BGR', bgr)

# cv.waitKey(0)

""" Blurring Techniques """
# img = basics.rescale(cv.imread("OpenCV/Data/Images/3.jpg"), 0.5)
# cv.imshow('Original', img)

# Averaging
# average = cv.blur(img, (3, 3))
# cv.imshow('Average', average)

# Gaussian Blur (less blur than average, but more natural)
# gaussian = cv.GaussianBlur(img, (3, 3), 0)
# cv.imshow('Gaussian', gaussian)

# Median Blur (reduces noise, but less effective than Gaussian)
# median = cv.medianBlur(img, 3)
# cv.imshow('Median', median)

# Bilateral Blur (most effective, retains edges)
# bilateral = cv.bilateralFilter(img, 10, 35, 25)         # 10 : diameter, 35 : colourInNeighbourhood, 25 : influenceSize
# cv.imshow('Bilateral', bilateral)

# cv.waitKey(0)

""" Bitwise Operations """
# blank = basics.createBlank(500, 500, 1)
# b1 = cv.rectangle(blank.copy(), (100, 100), (400, 400), 255, -1)
# b2 = cv.circle(blank.copy(), (250, 250), 175, 255, -1)
# cv.imshow('b1', b1)
# cv.imshow('b2', b2)

# Bitwise AND [ Intersection ]
# bitwiseAnd = cv.bitwise_and(b1, b2)
# cv.imshow('Bitwise AND', bitwiseAnd)

# Bitwise OR [ Union ]
# bitwiseOr = cv.bitwise_or(b1, b2)
# cv.imshow('Bitwise OR', bitwiseOr)

# Bitwise XOR [ Non-intersecting ]
# bitwiseXor = cv.bitwise_xor(b1, b2)
# cv.imshow('Bitwise XOR', bitwiseXor)

# Bitwise NOT [ Inverse ]
# bitwiseNot = cv.bitwise_not(b2)
# cv.imshow('Bitwise NOT', bitwiseNot)

# cv.waitKey(0)

""" Masking """
# img = basics.rescale(cv.imread("OpenCV/Data/Images/3.jpg"), 0.5)
# cv.imshow('Original', img)

# blank = basics.createBlank(img.shape[0], img.shape[1], 1)
# mask = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 69, 255, -1)
# cv.imshow('Blank', mask)

# masked = cv.bitwise_and(img, img, mask=mask)
# cv.imshow('Masked', masked)

# cv.waitKey(0)

""" Histograms """
# img = basics.rescale(cv.imread("OpenCV/Data/Images/3.jpg"), 0.5)
# cv.imshow('Original', img)

# Grayscale Histogram
# gray_hist = cv.calcHist([basics.gray(img)], [0], None, [256], [0, 256])
# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('No. of Pixels')
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.savefig('OpenCV/histogram.png')

# Colour Histogram
# plt.figure()
# plt.title('Colour Histogram')
# plt.xlabel('Bins')
# plt.ylabel('No. of Pixels')
# colors = ('b', 'g', 'r')
# for i, col in enumerate(colors):
#     hist = cv.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])
# plt.savefig('OpenCV/histogram.png')

# cv.waitKey(0)

""" Thresholding [ B & W ] """
# img = basics.gray(basics.rescale(cv.imread("OpenCV/Data/Images/3.jpg"), 0.5))
# cv.imshow('Original', img)

# Simple Thresholding
# _, thresh = cv.threshold(img, 128, 255, cv.THRESH_BINARY)               # cv.THRESH_BINARY_INV for inverse color
# cv.imshow('SimpleThreshold', thresh)

# Adaptive Thresholding
# adaptive = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 9)
# cv.imshow('AdaptiveThreshold', adaptive)

# cv.waitKey(0)

""" Edge Detection """
# img = basics.gray(basics.rescale(cv.imread("OpenCV/Data/Images/3.jpg"), 0.5))
# cv.imshow('Original', img)

# Canny Edge Detection
# canny = cv.Canny(img, 125, 175)
# cv.imshow('Canny', canny)

# Laplacian
# lap = cv.Laplacian(img, cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow('Laplacian', lap)

# Sobel
# sobelX = cv.Sobel(img, cv.CV_64F, 1, 0)
# sobelY = cv.Sobel(img, cv.CV_64F, 0, 1)
# combined = cv.bitwise_or(sobelX, sobelY)
# cv.imshow('Combined', combined)

# cv.waitKey(0)
