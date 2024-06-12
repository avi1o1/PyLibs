
import os
import cv2 as cv
import numpy as np

""" Defining Functions """
def rescale(img, scale=0.5):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

def gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

""" Face Detection """
# img = cv.imread("OpenCV/Data/Images/4.jpg")

# # Haar Cascade [ Not very accurate (Noise prone), but easy and fast ]
# data = cv.CascadeClassifier('OpenCV/Data/faceData.xml')
# faces = data.detectMultiScale(basics.gray(img), 1.1, 2)

# for (x1, y1, x2, y2) in faces:
#     cv.rectangle(img, (x1, y1), (x1+x2, y1+y2), (0, 0, 255), 2)

# cv.imshow('Detected Faces', img)
# cv.waitKey(0)

""" Image Detection in Video Feed """
# cap = cv.VideoCapture(0)
# data = cv.CascadeClassifier('OpenCV/Data/faceData.xml')

# while True:
#     ret, frame = cap.read()
#     faces = data.detectMultiScale(basics.gray(frame), 1.1, 2)

#     for (x1, y1, x2, y2) in faces:
#         cv.rectangle(frame, (x1, y1), (x1+x2, y1+y2), (0, 0, 255), 2)

#     cv.imshow('Detected Faces', frame)
#     if cv.waitKey(33) & 0xFF == ord('q'):
#         break

""" Face Recognition """
# Extracting and Labelling the training data (includes detection)
def extractData(dir):
    f, l = [], []
    l_ctr = 0
    haar = cv.CascadeClassifier('OpenCV/Data/faceData.xml')

    for person in os.listdir(dir):
        path = os.path.join(dir, person)
        for img in os.listdir(path):
            # Reading the image
            img_path = os.path.join(path, img)
            img_array = basics.gray(basics.rescale(cv.imread(img_path), 2))
            # cv.imshow('Image', img_array)
            # cv.waitKey(0)

            # Detecting the face
            detect = haar.detectMultiScale(img_array, 1.1, 3)
            for (x1, y1, x2, y2) in detect:
                face = img_array[y1:y1+y2, x1:x1+x2]
                f.append(face)
                l.append(l_ctr)
                # print(l_ctr, person)
        l_ctr += 1
    return f, l

# Training the Recognizer
def trainData(features, labels):
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)

    face_recognizer.save('OpenCV/Data/face_trained.yml')                    # Saving the trained data
    # np.save('OpenCV/Data/Features.npy', features)
    # np.save('OpenCV/Data/Labels.npy', labels)

# dir = 'OpenCV/Data/Images/trainFaces'
# features, labels = extractData(dir)
# # print(features, labels)
# trainData(np.array(features, dtype='object'), np.array(labels))

""" Testing the Recognizer """
def detectFace(img):
    haar = cv.CascadeClassifier('OpenCV/Data/faceData.xml')
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read('OpenCV/Data/face_trained.yml')
    # features = np.load('OpenCV/Data/Features.npy', allow_pickle=True)
    # labels = np.load('OpenCV/Data/Labels.npy')

    gray = basics.gray(img)
    people = [p for p in os.listdir('OpenCV/Data/Images/trainFaces')]

    faceRect = haar.detectMultiScale(gray, 1.1, 3)
    for (x1, y1, x2, y2) in faceRect:
        face = gray[y1:y1+y2, x1:x1+x2]
        label, _ = face_recognizer.predict(face)                            # _ (confidence) is inverse of accuracy

        cv.putText(img, people[label], (x1+13, y1+y2+23), cv.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 1)
        cv.rectangle(img, (x1, y1), (x1+x2, y1+y2), (0, 255, 0), 2)
        cv.imshow('Detected', img)
        cv.waitKey(0)

img = cv.imread('OpenCV/Data/Images/testFaces/find3.jpeg')
detectFace(img)

"""
Note : The built-in face recognizer (one we discussed above) is not very accurate, and is not recommended for
       large projects. Also, not only do we have a limited training dataset, but also the training data is not
       very accurate. So, it is recommended to use a more advanced and accurate face recognizer, like DNN.
"""
