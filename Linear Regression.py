from scipy.io import arff
import pandas as pd
import numpy as np
import csv

def generateDataSet(csvFile):
    dataSet = []
    dataSetLabels = []
    with open(csvFile, 'r') as csvfile:
        csvR = csv.reader(csvfile, delimiter = ',')
        for data in csvR:
            dataEntry = []
            for i in range(1, len(data)-1):
                dataEntry.append(float(data[i]))
            dataSetLabels.append([float(data[len(data)-1])])
            dataSet.append(dataEntry)
        dataSet = np.array(dataSet)
        X0s = np.ones((dataSet.shape[0],1))
        dataSet = np.concatenate((X0s, dataSet), axis=1)
        return dataSet, np.array(dataSetLabels)

dataSet, dataSetLabels = generateDataSet("Real estate.csv")

class LinearRegression:
    def __init__(self, dataSet, dataSetLabels):
        self.dataSet = dataSet
        self.dataSetLabels = dataSetLabels
        self.theta = []
    def train(self):
        self.theta = np.linalg.inv(self.dataSet.transpose().dot(self.dataSet)).dot(self.dataSet.transpose()).dot(self.dataSetLabels)
        # print(self.theta)
    def predict(self, testingData):
        predictedDataLabels = np.matmul(testingData, self.theta)
        return predictedDataLabels
    def calculateAccuracy(self, testingData, testingDataLabels):
        predictedDataLabels = self.predict(testingData)
        # print(predictedDataLabels.shape)
        predictedDataLabels = predictedDataLabels.reshape(predictedDataLabels.shape[0])
        testingDataLabels = testingDataLabels.reshape(testingDataLabels.shape[0])

        RSS = np.sum(np.square(testingDataLabels-predictedDataLabels))
        meanY = np.mean(predictedDataLabels)
        TSS = 0
        for z in testingDataLabels:
            TSS = TSS + (z-meanY)**2
        R2 = 1 -(RSS/TSS)
        print("R-squared is ", R2)
        print("Root mean squarred error is ", (RSS/testingDataLabels.shape[0])**0.5)

# print(dataSet.shape)
# print(dataSetLabels.shape)

trainingData = dataSet[:330, :]
testingData = dataSet[330:, :]
trainingDataLabels = dataSetLabels[:330, :]
testingDataLabels =  dataSetLabels[330:, :]
tester = LinearRegression(trainingData, trainingDataLabels)
tester.train()
tester.calculateAccuracy(testingData, testingDataLabels)