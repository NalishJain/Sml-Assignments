from scipy.io import arff
import pandas as pd
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def generateDataSet(csvFile):
    dataSet = []
    clusters = {1:[], 2:[], 3:[]}
    clusterIds = {"Iris-setosa" : 1, "Iris-versicolor" : 2, "Iris-virginica" : 3}
    with open(csvFile, 'r') as csvfile:
        csvR = csv.reader(csvfile, delimiter = ',')
        for data in csvR:
            dataEntry = []
            for i in range(1, len(data)-1):
                dataEntry.append(float(data[i]))
            dataEntry.append(int(clusterIds[data[len(data)-1]]))
            dataSet.append(dataEntry)
            clusters[int(clusterIds[data[len(data)-1]])].append(dataEntry[:4])
    return np.array(dataSet), clusters
dataSet1, clusters = generateDataSet("Iris.csv")

# print(dataSet)
# print(dataSet.shape)
# print(clusters[3], len(clusters[1]), len(clusters[2]))
class LDA:
    def __init__(self, dataSet, clusters, numDimensions):
        self.dataSet = dataSet
        self.numDimensions = numDimensions
        self.clusters = clusters
        self.Matrix = []
        self.sB = []
        self.sW = []
        self.clusterMeans = []
        self.overAllMean = []
        self.noOfPointsInCluster = []
        self.eValues = [] 
        self.eVectors = []
        self.reducedFeatures = []
    def calcualteMeans(self):  
        for i in clusters:
            clusters[i] = np.array(clusters[i])
            self.noOfPointsInCluster.append(len(clusters[i]))
            self.clusterMeans.append(np.mean(clusters[i], axis = 0))
        self.clusterMeans = np.array(self.clusterMeans)
        self.overAllMean = np.mean(self.dataSet[:, :4], axis = 0)          
    def calculateSW(self):
        self.sW = np.zeros((len(self.dataSet[0])-1, len(self.dataSet[0])-1))
        for i in clusters:
            for x in clusters[i]:
                xMinusMk = np.array([x - self.clusterMeans[i-1]])
                xMinusMkT = xMinusMk.transpose()
                self.sW += xMinusMkT.dot(xMinusMk)
    def calculateSB(self):
        self.sB = np.zeros((len(self.dataSet[0])-1, len(self.dataSet[0])-1))
        counter = 0
        for mK in self.clusterMeans:
            mkMinusM = np.array([mK-self.overAllMean])
            mkMinusMT = mkMinusM.transpose()
            self.sB +=(mkMinusMT*self.noOfPointsInCluster[counter]).dot(mkMinusM)
            counter += 1
    def eigenValueAndVector(self):
        self.Matrix = np.dot(np.linalg.inv(self.sW), self.sB)
        self.eValues, self.eVectors = np.linalg.eig(self.Matrix)
    def sortingEigenValues(self):  
        self.eValues = self.eValues[np.argsort(self.eValues)[::-1]]
        self.eVectors = self.eVectors[:, np.argsort(self.eValues)[::-1]]
    def reducingFeatures(self):
        for x in self.dataSet:
            c = np.matmul(x[:len(x)-1], self.eVectors[: self.numDimensions, :].transpose())
            c= np.append(c, x[len(x)-1])
            self.reducedFeatures.append(c)
        self.reducedFeatures = np.array(self.reducedFeatures)

    def algorithim(self):
        self.calcualteMeans()
        self.calculateSW()
        self.calculateSB()
        self.eigenValueAndVector()
        self.sortingEigenValues()
        self.reducingFeatures()
        

trainingData11 = dataSet1[:28, :]
trainingData12 = dataSet1[53:80, :]
trainingData13 = dataSet1[90:126, :]

trainingData = np.concatenate((trainingData11, trainingData12, trainingData13), axis=0)
trainingDataLabels = trainingData[:, dataSet1.shape[1]-1:dataSet1.shape[1]]
trainingDataLabels = trainingDataLabels.reshape(trainingDataLabels.shape[0])
trainingData = trainingData[:, :dataSet1.shape[1]-1]

testingData11 = dataSet1[28:53, :dataSet1.shape[1]-1]
testingData11Labels = dataSet1[28:53, dataSet1.shape[1]-1:dataSet1.shape[1]]
testingData11Labels = testingData11Labels.reshape(testingData11Labels.shape[0])
testingData12 = dataSet1[80:90, :dataSet1.shape[1]-1]
testingData12Labels = dataSet1[80:90, dataSet1.shape[1]-1:dataSet1.shape[1]]
testingData12Labels = testingData12Labels.reshape(testingData12Labels.shape[0])
testingData13 = dataSet1[126:150, :dataSet1.shape[1]-1]
testingData13Labels = dataSet1[126:150, dataSet1.shape[1]-1:dataSet1.shape[1]]
testingData13Labels = testingData13Labels.reshape(testingData13Labels.shape[0])

testingData = np.concatenate((testingData11, testingData12, testingData13), axis = 0)
testingDataLabels = np.concatenate((testingData11Labels, testingData12Labels, testingData13Labels), axis=0)


# trainingData = dataSet1[:120, :dataSet1.shape[1]-1]
# testingData = dataSet1[120:, :dataSet1.shape[1]-1]
# trainingDataLabels = dataSet1[:120, dataSet1.shape[1]-1:dataSet1.shape[1]]
# testingDataLabels = dataSet1[120:, dataSet1.shape[1]-1:dataSet1.shape[1]]

# trainingDataLabels = trainingDataLabels.reshape(trainingDataLabels.shape[0])
# testingDataLabels = testingDataLabels.reshape(testingDataLabels.shape[0])

knn5 = KNeighborsClassifier(n_neighbors = 5)
knn5.fit(trainingData, trainingDataLabels)
predictedDataLabels = knn5.predict(testingData)

print("Accuracy with k=5", accuracy_score(testingDataLabels, predictedDataLabels)*100)

tester = LDA(dataSet1, clusters, 2)
tester.algorithim()
dataSet1 = tester.reducedFeatures

# trainingData = dataSet1[:120, :dataSet1.shape[1]-1]
# testingData = dataSet1[120:, :dataSet1.shape[1]-1]
# trainingDataLabels = dataSet1[:120, dataSet1.shape[1]-1:dataSet1.shape[1]]
# testingDataLabels = dataSet1[120:, dataSet1.shape[1]-1:dataSet1.shape[1]]

# trainingDataLabels = trainingDataLabels.reshape(trainingDataLabels.shape[0])
# testingDataLabels = testingDataLabels.reshape(testingDataLabels.shape[0])
trainingData11 = dataSet1[:28, :]
trainingData12 = dataSet1[53:80, :]
trainingData13 = dataSet1[90:126, :]

trainingData = np.concatenate((trainingData11, trainingData12, trainingData13), axis=0)
trainingDataLabels = trainingData[:, dataSet1.shape[1]-1:dataSet1.shape[1]]
trainingDataLabels = trainingDataLabels.reshape(trainingDataLabels.shape[0])
trainingData = trainingData[:, :dataSet1.shape[1]-1]

testingData11 = dataSet1[28:53, :dataSet1.shape[1]-1]
testingData11Labels = dataSet1[28:53, dataSet1.shape[1]-1:dataSet1.shape[1]]
testingData11Labels = testingData11Labels.reshape(testingData11Labels.shape[0])
testingData12 = dataSet1[80:90, :dataSet1.shape[1]-1]
testingData12Labels = dataSet1[80:90, dataSet1.shape[1]-1:dataSet1.shape[1]]
testingData12Labels = testingData12Labels.reshape(testingData12Labels.shape[0])
testingData13 = dataSet1[126:150, :dataSet1.shape[1]-1]
testingData13Labels = dataSet1[126:150, dataSet1.shape[1]-1:dataSet1.shape[1]]
testingData13Labels = testingData13Labels.reshape(testingData13Labels.shape[0])

testingData = np.concatenate((testingData11, testingData12, testingData13), axis = 0)
testingDataLabels = np.concatenate((testingData11Labels, testingData12Labels, testingData13Labels), axis=0)
knn5 = KNeighborsClassifier(n_neighbors = 5)
knn5.fit(trainingData, trainingDataLabels)
predictedDataLabels = knn5.predict(testingData)

print("Accuracy after doing LDA")
print("Accuracy with k=5", accuracy_score(testingDataLabels, predictedDataLabels)*100)


