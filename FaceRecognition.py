import os
import cv2 as cv
import numpy as np
from FaceNet import *
from keras.applications.imagenet_utils import preprocess_input


class FaceRecognition:
    def __init__(self):
        protoTxt = "./faceDetectionConfigs/deploy.prototxt.txt"
        modelPath = "./faceDetectionConfigs/res10_300x300_ssd_iter_140000.caffemodel"
        self.databasePath = "./database"
        self.faceDetector = cv.dnn.readNetFromCaffe(protoTxt, modelPath)
        self.faceVerifier = getFaceNet("./weights/facenet_weights.h5")
        self.epsilon = 0.8
        self.frameShape = (160, 160)

    def detectFace(self, userIMG):
        (h, w) = userIMG.shape[:2]
        resizedFrame = cv.resize(userIMG, (300, 300))
        blob = cv.dnn.blobFromImage(resizedFrame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.faceDetector.setInput(blob)
        detections = self.faceDetector.forward()

        detectionConfs = []
        detectionBoxes = []
        for i in range(detections.shape[2]):
            currentConf = detections[0, 0, i, 2]
            currentBox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            detectionConfs.append(currentConf)
            detectionBoxes.append(currentBox.astype("int16"))

        maxConfIdx = np.argmax(detectionConfs)
        (startX, startY, endX, endY) = detectionBoxes[maxConfIdx]
        cv.rectangle(userIMG, (startX, startY), (endX, endY), (255, 255, 255), 1)
        return userIMG[startY:endY, startX:endX]

    def preProcessImg(self, userIMG):
        processedIMG = cv.resize(userIMG, self.frameShape)
        processedIMG = np.expand_dims(processedIMG, axis=0)
        processedIMG = preprocess_input(processedIMG)
        return processedIMG

    @staticmethod
    def compareVecs(vec1, vec2):
        normalized_vec1 = vec1 / np.sqrt(np.sum(np.multiply(vec1, vec1)))
        normalized_vec2 = vec2 / np.sqrt(np.sum(np.multiply(vec2, vec2)))
        distance = normalized_vec1 - normalized_vec2
        distance = np.sum(np.multiply(distance, distance))
        return np.sqrt(distance)

    def verifyFace(self, userIMG):
        featureVec, isSucces = self.encodeIMG(userIMG)
        minEps = 9999
        minEpsUser = ""
        if isSucces:
            for user in [os.path.join(self.databasePath, userName) for userName in os.listdir(self.databasePath)]:
                userVec = np.load(user)
                distance = self.compareVecs(featureVec, userVec)
                if distance < minEps:
                    minEps = distance
                    minEpsUser = user
        if minEps > self.epsilon:
            return "User not identified !"
        else:
            return os.path.basename(minEpsUser[:-4])

    def encodeIMG(self, userIMG):
        featureVec = None
        try:
            userFace = self.detectFace(userIMG)
            userFace = self.preProcessImg(userFace)
            featureVec = self.faceVerifier.predict(userFace)
            isSucces = True
        except:
            isSucces = False
        return featureVec, isSucces

