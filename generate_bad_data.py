from matplotlib import style

style.use("ggplot")
import csv
import json
import os
import pandas as pd
from sklearn.svm import SVR
import glob


class BadDataGenerator:

    def __init__(self, file_dir):
        print("generating bad data...")
        self.file_dir = file_dir

    def open_files(self):
        all_files = glob.glob(self.file_dir + "/*.csv")
        emp_list = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=0, header=0)
            emp_list.append(df)
            current_mean = df["Alt"].mean()
            print("mean of Alt for ", filename, " is ", df["Alt"].mean())
            dfupdate= df.sample(20)
            bad_alt = current_mean * 0.90
            dfupdate.Alt = bad_alt
            df.update(dfupdate)
            newfile = filename.split("\\")
            csvname = newfile[1].split("gp.")
            csvname_2 = csvname[0]+"dp"
            df.to_csv("./bad_auto/" + csvname_2 + ".csv")

generator = BadDataGenerator("./formatted_auto")
generator.open_files()

