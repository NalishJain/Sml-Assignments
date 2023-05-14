from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generateDataSet(csvFile):
    dataSet = []
    clusters = {0:[], 1:[]}
    clusterIds = {"Yes" : 0, "No" : 1}
    with open(csvFile, 'r') as csvfile:
        csvR = csv.reader(csvfile, delimiter = ',')
        for data in csvR:
            dataEntry = []
            for i in range(1, len(data)-1):
                if(data[i] =="normal"):
                    dataEntry.append(0)
                elif(data[i] == "reversable"):
                    dataEntry.append(1)
                elif(data[i] == "fixed"):
                    dataEntry.append(2)
                elif(data[i] == "asymptomatic"):
                    dataEntry.append(0)
                elif(data[i] == "nonanginal"):
                    dataEntry.append(1)
                elif(data[i] == "nontypical"):
                    dataEntry.append(2)
                elif(data[i] == "typical"):
                    dataEntry.append(3)
                elif(data[i] == "NA"):
                    dataEntry.append(1000)
                else:
                    dataEntry.append(float(data[i]))
            dataEntry.append(int(clusterIds[data[len(data)-1]]))
            dataSet.append(dataEntry)
            clusters[int(clusterIds[data[len(data)-1]])].append(dataEntry[:len(dataEntry)-1])
    clusters[0] = np.array(clusters[0])
    clusters[1] = np.array(clusters[1])
    return np.array(dataSet), clusters

class FDA:
    def __init__(self, dataSet, clusters):
        self.dataSet = dataSet
        self.clusters = clusters
        self.Matrix = []
        self.clusterMeans = []
        self.noOfPointsInCluster = []
        self.reducedFeatures = []
        self.s0 = []
        self.s1 = []
        self.omega = []
        self.transfromedData = []
    def calcualteMeans(self):  
        for i in self.clusters:
            self.noOfPointsInCluster.append(len(self.clusters[i]))
            self.clusterMeans.append(np.mean(self.clusters[i], axis = 0))
        self.clusterMeans = np.array(self.clusterMeans)
    def calculateS0ansS1(self):
        self.s0 = np.zeros((len(self.dataSet[0])-1, len(self.dataSet[0])-1))
        self.s1 = np.zeros((len(self.dataSet[0])-1, len(self.dataSet[0])-1))
        for x in self.clusters[0]:
            xMinusMk = np.array([x - self.clusterMeans[0]])
            xMinusMkT = xMinusMk.transpose()
            self.s0 += xMinusMkT.dot(xMinusMk)
        self.s0 = self.s0/(self.noOfPointsInCluster[0]-1)
        for x in self.clusters[1]:
            xMinusMk = np.array([x - self.clusterMeans[1]])
            xMinusMkT = xMinusMk.transpose()
            self.s1 += xMinusMkT.dot(xMinusMk)
        self.s1 = self.s1/(self.noOfPointsInCluster[1]-1)
        self.s0 = np.array(self.s0)
        self.s1 = np.array(self.s1)
    def calculateOmega(self):
        self.Matrix = np.linalg.inv(self.noOfPointsInCluster[0]*self.s0 + self.noOfPointsInCluster[1]*self.s1)
        meanDifference = np.array(self.clusterMeans[0] - self.clusterMeans[1])
        self.omega = np.matmul(self.Matrix, meanDifference.transpose())
    def calculateTransfromedData(self):
        for x in self.dataSet:
            c = np.matmul(x[:len(x)-1], self.omega)
            c= np.append(c, int(x[len(x)-1]))
            # c= np.append(c, 1)
            self.transfromedData.append(c)
        self.transfromedData = np.array(self.transfromedData)
    def algorithim(self):
        self.calcualteMeans()
        self.calculateS0ansS1()
        self.calculateOmega()
        self.calculateTransfromedData()

class logisticRegression:
    def __init__(self, dataSet, learningRate):
        self.dataSet = dataSet
        self.noOfSamples = dataSet.shape[0]
        self.noOfFeatures = dataSet.shape[1]
        self.learningRate = learningRate
        self.Yis = []
        self.Xis = []
    def sigmoidFunction(self, x):
        return 1/(1 + np.exp(-x))
    def train(self):
        self.theta = np.ones((self.noOfFeatures, 1))
        X0s = np.ones((self.noOfSamples,1))
        self.Yis = self.dataSet[:,self.noOfFeatures-1:self.noOfFeatures]
        self.Xis = self.dataSet[:, :self.noOfFeatures-1]
        # self.Yis = np.reshape(np.array(self.Yis), (1, self.noOfSamples))
        self.Xis = np.array(self.Xis)
        Xmean = self.Xis.mean(axis=0)
        Xstd = self.Xis.std(axis=0)
        # NormalisingData
        for z in range(self.Xis.shape[0]):
            self.Xis[z] = (self.Xis[z]-Xmean)/Xstd
        self.Xis = np.concatenate((X0s, self.Xis), axis=1)


        # Normalise your data
        for i in range(1000):
            Pis = []
            for z in self.Xis:
                Pis.append(self.sigmoidFunction(np.matmul(z, self.theta)))
            Pis = np.array(Pis).reshape((self.noOfSamples,1))
            self.theta = self.theta - (self.learningRate)*(1/self.noOfSamples)*np.dot(self.Xis.transpose(), Pis - self.Yis)
    def predict(self, dataSet):              
        X0s = np.ones((dataSet.shape[0],1))
        Xmean = dataSet.mean(axis=0)
        Xstd = dataSet.std(axis=0)
        for z in range(dataSet.shape[0]):
            dataSet[z] = (dataSet[z]-Xmean)/Xstd
        dataSet = np.concatenate((X0s, dataSet), axis=1)
        predictedPis = []
        for z in dataSet:
            predictedPis.append(self.sigmoidFunction(np.matmul(z, self.theta)))
        predYis = []
        predYis = [0 if Pi <= 0.5 else 1 for Pi in predictedPis]
        
        return np.array(predYis)
    
    def calculateAccuracy(self, dataSet, testingDataLabels):
        predYis = self.predict(dataSet)
        accuracy = np.sum(testingDataLabels == predYis) / len(testingDataLabels)
        print("Accurracy obtained is ", accuracy)

    


dataSet, clusters = generateDataSet("Heart.csv")

# trainingData = dataSet[:240, :]
trainingData1 = dataSet[:200, :]


validationData = dataSet[200:260, :dataSet.shape[1]-1]
validationDataLabels = dataSet[200:260, dataSet.shape[1]-1 : dataSet.shape[1]]
validationDataLabels = validationDataLabels.reshape(validationDataLabels.shape[0])



tester = logisticRegression(trainingData1, 0.6)
tester.train()

print("ONLY LOGISTIC REGRESSION")
print("Validation Accuracy")
tester.calculateAccuracy(validationData, validationDataLabels)

print("Training Accuracy ")
trainingData1 = dataSet[:200, :dataSet.shape[1]-1]
trainingData1Labels = dataSet[:200, dataSet.shape[1]-1 : dataSet.shape[1]]
trainingData1Labels = trainingData1Labels.reshape(trainingData1Labels.shape[0])

tester.calculateAccuracy(trainingData1, trainingData1Labels)

print("Testing Accuracy ")

testingData = dataSet[260:303, :dataSet.shape[1]-1]
testingDataLabels = dataSet[260:303, dataSet.shape[1]-1 : dataSet.shape[1]]
testingDataLabels = testingDataLabels.reshape(testingDataLabels.shape[0])

tester.calculateAccuracy(testingData, testingDataLabels)


print("FDA + LOGISTIC REGRESSION")
#USING FDA
dataTester = FDA(dataSet, clusters)
dataTester.algorithim()
newDataSet = dataTester.transfromedData

# TRAINING the model
trainingData = newDataSet[:200, :]
LRtester = logisticRegression(trainingData, 0.6)
LRtester.train()

# TRAINING DATA
print("Training Accuracy")
trainingData1 = newDataSet[:200, :newDataSet.shape[1]-1]
trainingData1Labels = newDataSet[:200, newDataSet.shape[1]-1 : newDataSet.shape[1]]
trainingData1Labels = trainingData1Labels.reshape(trainingData1Labels.shape[0])
LRtester.calculateAccuracy(trainingData1, trainingData1Labels)

# VALIDATION DATA
print("Validation Accuracy")

validationData = newDataSet[200:260, :newDataSet.shape[1]-1]
validationDataLabels = newDataSet[200:260, newDataSet.shape[1]-1 : newDataSet.shape[1]]
validationDataLabels = validationDataLabels.reshape(validationDataLabels.shape[0])
LRtester.calculateAccuracy(validationData, validationDataLabels)

# TESTING DATA
print("Testing Accuracy ")

testingData = newDataSet[260:303, :newDataSet.shape[1]-1]
testingDataLabels = newDataSet[260:303, newDataSet.shape[1]-1 : newDataSet.shape[1]]
testingDataLabels = testingDataLabels.reshape(testingDataLabels.shape[0])
LRtester.calculateAccuracy(testingData, testingDataLabels)

# Now using PCA
print("PCA + FDA + LOGISTIC REGRESSION")

xData = dataSet[:, :dataSet.shape[1]-1]
yData =  dataSet[:, dataSet.shape[1]-1 : dataSet.shape[1]]
sc = StandardScaler()
xData = sc.fit_transform(xData)
pca = PCA(n_components = 6)

xData = pca.fit_transform(xData)
pcaDataSet = np.concatenate((xData, yData), axis=1)
# ----
# Generating clusters again
newClusters = {0 : [], 1 :[] }
for z in pcaDataSet:
    newClusters[int(z[-1])].append(z[:len(z)-1])

newClusters[0] = np.array(newClusters[0])
newClusters[1] = np.array(newClusters[1])


dataTester = FDA(pcaDataSet, newClusters)
dataTester.algorithim()
newDataSet = dataTester.transfromedData
# print("-----")
# TRAINING the model
trainingData = newDataSet[:200, :]
LRtester = logisticRegression(trainingData, 0.6)
LRtester.train()

# TRAINING DATA
print("Training Accuracy")
trainingData1 = newDataSet[:200, :newDataSet.shape[1]-1]
trainingData1Labels = newDataSet[:200, newDataSet.shape[1]-1 : newDataSet.shape[1]]
trainingData1Labels = trainingData1Labels.reshape(trainingData1Labels.shape[0])
LRtester.calculateAccuracy(trainingData1, trainingData1Labels)

# VALIDATION DATA
print("Validation Accuracy")

validationData = newDataSet[200:260, :newDataSet.shape[1]-1]
validationDataLabels = newDataSet[200:260, newDataSet.shape[1]-1 : newDataSet.shape[1]]
validationDataLabels = validationDataLabels.reshape(validationDataLabels.shape[0])
LRtester.calculateAccuracy(validationData, validationDataLabels)

# TESTING DATA
print("Testing Accuracy ")

testingData = newDataSet[260:303, :newDataSet.shape[1]-1]
testingDataLabels = newDataSet[260:303, newDataSet.shape[1]-1 : newDataSet.shape[1]]
testingDataLabels = testingDataLabels.reshape(testingDataLabels.shape[0])
LRtester.calculateAccuracy(testingData, testingDataLabels)



