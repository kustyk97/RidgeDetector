import numpy as np
import cv2 as cv

class Map:

    def __init__(self, path):
        file = open(path, "r")

        self.cols = int(str.split(file.readline(), " ")[1])
        self.rows= int(str.split(file.readline(), " ")[1])
        self.xllcenter = float(str.split(file.readline(), " ")[1])
        self.yllcenter = float(str.split(file.readline(), " ")[1])
        self.cellsize = float(str.split(file.readline(), " ")[1])
        self.nodata_value = int(str.split(file.readline(), " ")[1])

        datas = file.readlines()
        self.data = np.empty([self.rows, self.cols])
        for i, row in enumerate(datas):
            dataList = str.split(row,)
            dataList = list(map(float, dataList))
            vector = np.array(dataList)
            self.data[i] = vector

        self.data[self.data == self.nodata_value] = np.nan
        self.nodata_value = np.nan

        
            
    def SetCellSize(self, newCellSize):
        scale = newCellSize/self.cellsize
        self.data = cv.resize(self.data, (int(self.rows/scale), int(self.cols/scale)), interpolation=cv.INTER_AREA)
        self.cellsize = newCellSize
        self.rows, self.cols = self.data.shape

    def SetMapSize(self, size):
        self.data = cv.resize(self.data, (size), interpolation=cv.INTER_AREA)
        self.rows, self.cols = self.data.shape

    def SetMapFromData(self, data, cellsize):
        self.data = data
        self.rows, self.cols = data.shape
        self.xllcenter = 0
        self.yllcenter = 0
        self.cellsize = cellsize
        self.nodata_value = np.nan

    def GetData(self):
        return self.data
    
    def GetCellSize(self):
        return self.cellsize
    


