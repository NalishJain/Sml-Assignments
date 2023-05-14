



import numpy as np
import random;
from matplotlib import pyplot as plt
import random;
# from google.colab.patches import cv2_imshow
import cv2
# from PIL importpi Image
import os
# from scipy import stats as st

class PCA:
  def __init__(self, features):
    self.noOfFeatures = features
    self.dataSet = []
    self.standardizedDataset = []
    self.cMatrix = []
    self.eValues = []
    self.eVectors = []
    self.reducedFeatures = []
    self.record = []
  def createDataset(self):
    DataSet = []
    DigitDataSet = {}
    for digitNumber in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
      DigitDataSet[int(digitNumber)] = 0
      for filename in os.listdir('C:/Users/hp/Desktop/trainingSet/'+ digitNumber):
          if filename.endswith("jpg"): 
            image = cv2.imread('C:/Users/hp/Desktop/trainingSet/' + digitNumber +'/' + filename)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            DataEntery = []
            for i in range(gray_image.shape[0]):
              DataEntery.extend(gray_image[i])
            # print(DataEntery)
            DataSet.append(DataEntery)
            DigitDataSet[int(digitNumber)] += 1
            # DigitDataSet[int(digitNumber)].append(DataEntery)
      

    # print(DataSet)
    self.dataSet = np.array(DataSet)
    self.record = DigitDataSet
  def standardization(self):
    columnMeans = np.mean(self.dataSet , axis = 0)
    # columnVariances = np.var(self.dataSet, axis = 0)
    # if(columnVariances == 0):
    #   print("0 found")
    # columnStandardDeviation = columnVariances**0.5
    self.standardizedDataset = self.dataSet - columnMeans
    self.standardizedDataset =  np.divide(self.standardizedDataset, np.std(self.dataSet, axis = 0))
  def covarianceMatrix(self):
    self.cMatrix = np.cov(self.standardizedDataset, rowvar = False)
  def eigenValueAndVector(self):
      self.eValues, self.eVectors = np.linalg.eig(self.cMatrix)
  def sortingEigenValues(self):  
    self.eValues = self.eValues[np.argsort(self.eValues)[::-1]]
    self.eVectors = self.eVectors[:, np.argsort(self.eValues)[::-1]]
  def reducingFeatures(self):
    self.reducedFeatures = np.dot((self.eVectors[:, 0:self.noOfFeatures]).transpose(),self.standardizedDataset.transpose()).transpose()

  def algorithim(self):
    self.createDataset()
    self.standardization()
    self.covarianceMatrix()
    self.eigenValueAndVector()
    self.sortingEigenValues()
    self.reducingFeatures()
  def plot(self):
    inp = []
    oup = []
    # print(self.eValues)
    flag = 0
    for i in range(len(self.eValues)):
      inp.append(i+1)
      # np.cumsum(self.eValues)
      if(sum(self.eValues[:i+1])/sum(self.eValues) >= 0.8 and flag == 0):
        flag = 1
        print("Number of features required are : ", i + 1)
    
      oup.append(sum(self.eValues[:i+1])/sum(self.eValues))
    # print(inp)
    # print(oup)
    plt.title("Graph")
    plt.ylabel("Explained-variance")
    plt.xlabel("Number of PCs")
    plt.plot(inp, oup, color = "red")
    plt.show()
import numpy as np
class Knn:
  def __init__(self, k, dataSet, maxLabelIndex, testSample):
    self.k = k
    self.dataSet = dataSet
    self.maxLabelIndex = maxLabelIndex
    self.testSample = testSample
  def calculateDistance(self, a, b):
    distance = 0
    for i in range(self.maxLabelIndex):
      distance = distance + abs(a[i] - b[i])
    return distance
  def algorithim(self):
    distanceSet = []
    for i in range(len(self.dataSet)):
      distanceSet.append((self.calculateDistance(self.dataSet[i], self.testSample), i))
    sortedDistances = sorted(distanceSet)
    # print(sortedDistances)
    # Picking k nearest element
    kNearest = []
    for i in sortedDistances[:self.k]:
      kNearest.append(self.dataSet[i[1]])
    kNearest = np.array(kNearest)
    return kNearest

def preparingData(noOfFeatures):
  print("Entered preparingData")
  test = PCA(noOfFeatures)
  test.algorithim()
  print("Shape is",test.reducedFeatures.shape)
  # test.plot()
  trainingDataSet = []
  testingDataSet = []
  iteraror = 0   
  print(test.record)
  for digit in test.record:
    # print(digit)
    # print(test.record[digit])
    trainigData = int(test.record[digit]*0.8)
    # print("Training size for digit : ", digit, " is ", trainigData)
    for index in range(trainigData):
      # print(digit)
      l = test.reducedFeatures[iteraror]
      # l.append(digit)
      l = np.append(l, digit)
      trainingDataSet.append(l)
      # print(trainingDataSet[iteraror])
      iteraror += 1
      # print(iteraror)
    for index in range(trainigData, test.record[digit]):
      l = test.reducedFeatures[iteraror]
      l = np.append(l, digit)
      testingDataSet.append(l)
      iteraror += 1 
      print(iteraror) 
  print("Exited preparingData")
  # for i in trainingDataSet:
  #   print("last element is ", i[ -1])
  return trainingDataSet, testingDataSet

def calculateMode(kNearest):
  labels = []
  # print(kNearest)
  for element in kNearest:
    labels.append(element[-1])
  labels.sort()
  max1 = 0
  tmp = 1
  mode = 0
  for i in range(len(labels) - 1):
    if(labels[i] == labels[i+1]):
      tmp += 1
    else:
      tmp = 1
    if(tmp >= max1):
      max1 = tmp
      mode = labels[i]
  return mode

  
def checkingAccuracy(noOfFeatures):
  print("Entered checking Accuracy")
  trainingDataSet, testingDataSet = preparingData(noOfFeatures)
  Accuracy = 0
  counter =0 
  print("About to checked accuracy")
  for i in testingDataSet:
    tester = Knn(3, trainingDataSet, noOfFeatures, i)
    answer1 = tester.algorithim()
    answer = calculateMode(answer1)
    print(counter, " Point checked")
    # print(answer, i[-1])
    counter += 1
    if(answer == i[-1]):
      Accuracy += 1
  print("Accuracy is ", Accuracy/len(testingDataSet))

checkingAccuracy(25)

# REPORT
# Classification accuracy without pca and k = 10 is 0.9860797
# Classification accuracy for pca = 5 and k = 3 is 0.724330755502677
# Classification accuracy for pca = 25 and k = 3 is  0.890789467938123
# Classification accuracy for pca = 125 and k = 3 is 0.9601546698393813

# Pcs needed for 80% variance are 187
