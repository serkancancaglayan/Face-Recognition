import cv2
from DataBase import *
from FaceRecognition import *
import argparse
from sys import argv

if __name__ == '__main__':
    dBase = DataBase()
    faceRecognition = FaceRecognition()
    if len(argv) > 0:
        argParser = argparse.ArgumentParser()
        argParser.add_argument('--videoSource', type=str, help="Webcam source", default="0")
        argParser.add_argument('--operation', type=str, help="Create / Delete User")
        argParser.add_argument('--userName', type=str, help="User Name")
        args = argParser.parse_args()
        source = 0 if args.videoSource == "0" else args.videoSource
        operation = args.operation
        userName = args.userName

        if operation == "n":
            videoCap = cv2.VideoCapture(source)
            while True:
                ret, frame = videoCap.read()
                cv.putText(frame, "Press Q button to create user, ESC to exit.", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 255, 255), 2, cv.LINE_AA)
                cv.imshow("Create User", frame)
                pressedKey = cv.waitKey(1)
                if pressedKey == ord('q'):
                    dBase.createUser(userName, frame, faceRecognition.encodeIMG)
                elif pressedKey == 27:
                    break
            videoCap.release()
            cv.destroyAllWindows()

        elif operation == "d":
            dBase.deleteUser(userName)

        else:
            print('Invalid operation <n , d>')

