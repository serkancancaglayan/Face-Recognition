import cv2 as cv
import queue
from FaceRecognition import FaceRecognition
import threading

frameQueue = queue.Queue(maxsize=30)
faceRecognition = FaceRecognition()
videoCapture = cv.VideoCapture(0)


def putQueue():
    while True:
        ret, frame = videoCapture.read()
        if ret:
            frameQueue.put(frame)


def getQueue():
    while True:
        if not frameQueue.empty():
            frame = frameQueue.get()
            userID = faceRecognition.verifyFace(frame)
            cv.putText(frame, userID, (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (255, 255, 255), 2, cv.LINE_AA)
            cv.imshow("frame", frame)
            if cv.waitKey(1) == 27:
                videoCapture.release()
                cv.destroyAllWindows()
                break


if __name__ == "__main__":
    videoCapThread = threading.Thread(target=putQueue, daemon=True).start()
    faceVerifierThread = threading.Thread(target=getQueue).start()
