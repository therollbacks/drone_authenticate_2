import csv
import random
import math
import operator
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class kNearest:

    def __init__(self):

        trainingSet = []
        testSet = []
        self.split = 0.67
        self.loadDataset('./compared/001compared.csv', self.split, trainingSet, testSet)
        print(('Train set: ' + repr(len(trainingSet))))
        print(('Test set: ' + repr(len(testSet))))
        # generate predictions
        predictions = []
        k = 3
        for x in range(len(testSet)):
            neighbors = self.getNeighbors(trainingSet, testSet[x], k)
            result = self.getResponse(neighbors)
            predictions.append(result)
            print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
        accuracy = self.getAccuracy(testSet, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')


    def loadDataset(self, filename, split, trainingSet=[], testSet=[]):
        """
        loads dataset
        :param filename:
        :param split:
        :param trainingSet:
        :param testSet:
        :return:
        """
        with open(filename, 'r') as csvfile:
            next(csvfile)
            lines = csv.reader(csvfile)
            dataset = list(lines)
            for x in range(len(dataset) - 1):
                for y in range(4):
                    dataset[x][y] = float(dataset[x][y])
                if random.random() < self.split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])

    def euclideanDistance(self, instance1, instance2, length):
        """
        calculates euclidean distance
        :param instance1:
        :param instance2:
        :param length:
        :return:
        """
        distance = 0
        for x in range(length):
            distance += pow((float(instance1[x]) - float(instance2[x])), 2)
        return math.sqrt(distance)

    def getNeighbors(self, trainingSet, testInstance, k):
        """
        find distances from neighbours
        :param trainingSet:
        :param testInstance:
        :param k:
        :return:
        """
        distances = []
        length = len(testInstance) - 1
        for x in range(len(trainingSet)):
            dist = self.euclideanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def getResponse(self, neighbors):
        """
        calculates votes from neighbours
        :param neighbors:
        :return:
        """
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def getAccuracy(self, testSet, predictions):
        """
        calculates performance matrix
        :param testSet:
        :param predictions:
        :return:
        """
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct += 1

        TP = FN = FP = TN = 0
        for j in range(len(testSet)):

            if testSet[j][-1] == 0 and predictions[j] == 0:
                TP = TP + 1
            elif testSet[j][-1] == 0 and predictions[j] == 1:
                FN = FN + 1
            elif testSet[j][-1] == 1 and predictions[j] == 0:
                FP = FP + 1
            else:
                TN = TN + 1

        print(TP, FN, FP, TN)

        accuracy = (TP + TN) / (TP + FN + FP + TN)
        print("accuracy:", accuracy)
        # sensitivity = TP / (TP + FN)
        # print('sensitivity: ', sensitivity)
        specificity = TN / (TN + FP)
        print('specifitiy: ', specificity)
        return (correct / float(len(testSet))) * 100.0


if __name__ == '__main__': kNearest()
