import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.pyplot as plt
import glob,os

class LogisticRegression:

    def __init__(self, filename):
        self.filename = filename

    def algorithm(self):
        my_counter = 0
        # data = np.loadtxt('./compared_auto_backup/cleandatagp4_010compared.csv', delimiter=",", skiprows=1)
        # for file in file_list:
        data = np.loadtxt(self.filename, delimiter=",", skiprows=1)
        x = data[:, 5:7]
        print(x)
        y = data[:, 7]

        X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
        X[:, 1:] = x

        initial_theta = np.zeros(X.shape[1])  # set initial model parameters to zero
        theta = opt.fmin_cg(self.cost, initial_theta, self.cost_gradient, (X, y))

        # x_axis = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
        # y_axis = (-1 / theta[2]) * (theta[1] * x_axis + theta[0])
        # ax.plot(x_axis, y_axis, linewidth=2)

        predictions = np.zeros(len(y))
        predictions[self.sigmoid(X @ theta) >= 0.5] = 1
        TP = FN = FP = TN = 0
        # print(len(y), len(predictions))
        for j in range(len(y)):
            if y[j] == 0 and predictions[j] == 1:
                TP = TP + 1
            elif y[j] == 0 and predictions[j] == -1:
                FN = FN + 1
            elif y[j] == 1 and predictions[j] == 1:
                FP = FP + 1
            else:
                TN = TN + 1
        # print(TP, FN, FP, TN)

        # Performance Matrix

        accuracy = (TP + TN) / (TP + FN + FP + TN)

        accuracy = (TP + TN) / (TP + FN + FP + TN)
        print("accuracy:", accuracy)
        try:
            precision = TP / (TP + FP)
            print("precision: ", precision)
        except:
            precision = 1
        try:
            sensitivity = TP / (TP + FN)
            print('recall: ', sensitivity)
        except:
            sensitivity = 1
        specificity = TN / (TN + FP)
        print('specifitiy: ', specificity)
        try:
            f1 = (2 * (precision * sensitivity)) / (precision + sensitivity)
            print('f1: ', f1)
        except:
            f1 = 1
        return accuracy, specificity, sensitivity, precision, f1


        #avg_spec_list.append(specificity)


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, theta, X, y):
        predictions = self.sigmoid(X @ theta)
        predictions[predictions == 1] = 0.999  # log(1)=0 causes division error during optimization
        error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
        return sum(error) / len(y)

    def cost_gradient(self, theta, X, y):
        predictions = self.sigmoid(X @ theta)
        return X.transpose() @ (predictions - y) / len(y)

    @staticmethod
    def find_ave(list):
        tmp = 0
        for each in list:
            tmp = tmp + each
        return tmp/len(list)


os.chdir("./compared")
acc_list = []
sensitivity_list = []
specificity_list = []
prec_list = []
f1_list = []


for file in glob.glob("*.csv"):
    log_res = LogisticRegression(file)

    acc,spec, sens, prec, f1 = log_res.algorithm()
    acc_list.append(acc)
    specificity_list.append(spec)
    sensitivity_list.append(sens)
    prec_list.append(prec)
    f1_list.append(f1)


print(acc_list)
print("-----------------------------------------------------------------")
print('average accuracy = ', LogisticRegression.find_ave(acc_list))
print('average specificity = ', LogisticRegression.find_ave(specificity_list))
print('average recall = ', LogisticRegression.find_ave(sensitivity_list))
print('average precision = ', LogisticRegression.find_ave(prec_list))
print('average f1 = ', LogisticRegression.find_ave(f1_list))