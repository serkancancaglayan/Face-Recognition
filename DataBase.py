import numpy as np
import os


class DataBase:
    def __init__(self):
        self.dataBasePath = "./database"
        isExist = os.path.exists(self.dataBasePath)
        if not isExist:
            os.mkdir(self.dataBasePath)

    def createUser(self, usrName, usrIMG, encodeFunction):
        featureVec, isSucceed = encodeFunction(usrIMG)
        np.save(self.dataBasePath + '/' + usrName + '.npy', featureVec)
        if isSucceed == 0:
            print('Could not save user {} to database !'.format(usrName))
        else:
            print('User {} saved successfully to database !'.format(usrName))
        return isSucceed

    def deleteUser(self, usrName):
        userPath = self.dataBasePath + '/' + usrName + '.npy'
        isExist = os.path.exists(userPath)
        if isExist:
            os.remove(userPath)
            print("User {} deleted successfully !".format(usrName))
        else:
            print("User {} does not exist in database !".format(usrName))
