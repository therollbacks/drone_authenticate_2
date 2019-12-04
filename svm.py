import pandas as pd
import numpy as np
from sklearn import svm
import os, glob

class SVM:

    def __init__(self, filename):
        # cc = pd.read_csv("./bad_auto/cleandatagp3_047compared.csv")
        self.filename = filename
        self.train_set_len = 90
        self.model()

    def model(self):

        cc = pd.read_csv(self.filename)

        # .loc primarily label based. used to access a column data or row
        nor_obs = cc.loc[cc.Category == 0]  # Data frame with normal observation
        ano_obs = cc.loc[cc.Category == 1]

        # training set: train_features
        # test observations/features: X_test
        # test_labels: Y_test

        train_feature = nor_obs.loc[0:self.train_set_len, :]
        train_feature = train_feature.drop('Category', 1)

        # Y_1 is all the "category" value after and including row 300
        Y_1 = nor_obs.loc[self.train_set_len:, 'Category']

        # Y_2 is all the rows that have category ==1
        Y_2 = ano_obs['Category']

        # Creatng test observations/features

        # from 200000 till last row, drop category == 1
        X_test_1 = nor_obs.loc[self.train_set_len:, :].drop('Category', 1)

        # x test 2 drops all the rows where category == 1
        X_test_2 = ano_obs.drop('Category', 1)

        # append both
        X_test = X_test_1.append(X_test_2)

        # setting hyper parameters for once class svm
        oneclass = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)

        # y_1 get all the data where after row 200 000

        Y_test = Y_1.append(Y_2)

        oneclass.fit(train_feature)
        fraud_pred = oneclass.predict(X_test)
        # print('fraud pred is ', fraud_pred)

        # Check the number of outliers predicted by the algorithm
        unique, counts = np.unique(fraud_pred, return_counts=True)
        # print(np.asarray((unique, counts)).T)

        # Convert Y-test and fraud_pred to dataframe for ease of operation
        # makes a table using Y test
        Y_test = Y_test.to_frame()

        #creates index
        Y_test = Y_test.reset_index()
        # print(Y_test)

        # print('y_test with index is ', Y_test)
        # creates dataframe using fraud pred (list of 139)
        fraud_pred = pd.DataFrame(fraud_pred)

        # print('fraud pred data frame ', fraud_pred)
        fraud_pred = fraud_pred.rename(columns={0: 'prediction'})

        # if not a fraud
        fraud_pred[fraud_pred['prediction']==1]=0
        # if a fraud

        fraud_pred[fraud_pred['prediction']==-1]=1

        # print(fraud_pred['prediction'].value_counts())
        # print(sum(fraud_pred['prediction'])/fraud_pred['prediction'].shape[0])
        countt = 0
        for i in Y_test['Category']:
            if i == 1:

                countt += 1
        print('Number of ones in csv file:',countt)

        ##Performance check of the model


        TP = FN = FP = TN = 0
        for j in range(len(Y_test)):
            if Y_test['Category'][j] == 0 and fraud_pred['prediction'][j] == 1:
                TP = TP + 1
            elif Y_test['Category'][j] == 0 and fraud_pred['prediction'][j] == -1:
                FN = FN + 1
            elif Y_test['Category'][j] == 1 and fraud_pred['prediction'][j] == 1:
                TN = TN + 1
            else:
                FP = FP + 1
        print(TP, FN, FP, TN)

        # Performance Matrix

        accuracy = (TP + TN) / (TP + FN + FP + TN)
        print("accuracy:", accuracy)
        sensitivity = TP / (TP + FN)
        print('sensitivity: ', sensitivity)
        specificity = TN / (TN + FP)
        print('specifitiy: ', specificity)

        return (TP, FN, FP, TN, accuracy, sensitivity, specificity)

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

for file in glob.glob("*.csv"):
    svm_model = SVM(file)
    TP, FN, FP, TN, accuracy, sensitivity, specificity= svm_model.model()
    acc_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)

print("-----------------------------------------------------------------")
print('average accuracy = ', SVM.find_ave(acc_list))
print('average sensitivity = ', SVM.find_ave(sensitivity_list))
print('average specificity = ', SVM.find_ave(specificity_list))
