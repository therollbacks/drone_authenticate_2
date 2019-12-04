import glob, os
from matplotlib import style
import csv
from sklearn.svm import SVR
from numpy import linspace
import math

style.use("ggplot")

svr = SVR(kernel="linear", gamma="auto")

# go through both files and find dissimilar longitude and latitude

headers = ['Index', 'Alt', 'Lat', 'Lng', 'Yaw(deg)', 'Roll(deg)', 'Pitch(deg)']
first_line = ['Index', 'Alt', 'Lat', 'Lng', 'Yaw(deg)', 'Roll(deg)', 'Pitch(deg)', 'Category']
full_data = []
sample_size = 100000
filename_list = []


class TestCsv:

    def __init__(self):
        self.true_lat_list = []
        self.true_lng_list = []
        self.true_roll_list = []
        self.true_pitch_list = []
        self.true_yaw_list = []
        self.false_lat_list = []
        self.false_lng_list = []
        self.lat_min = 0
        self.lng_min = 0
        self.lat_max = 0
        self.lng_max = 0

    # true
    def open_file_w_headers(self, goodfile, badfile, justname):

        self.true_lat_list = []
        self.true_lng_list = []
        self.true_roll_list = []
        self.true_pitch_list = []
        self.true_yaw_list = []
        self.true_alt_list = []

        self.false_lat_list = []
        self.false_lng_list = []

        self.lat_min = 0
        self.lng_min = 0
        self.lat_max = 0
        self.lng_max = 0

        with open("../formatted_auto/" + goodfile) as f:
            print("goodfile is ", goodfile)
            print("badfile is ", badfile)

            next(f)
            for line in csv.reader(f):
                true_lat = format(float(line[2]), '.5f')
                true_lng = format(float(line[3]), '.5f')
                true_roll = format(float(line[5]), '.1f')
                true_pitch = format(float(line[6]), '.1f')
                true_alt = format(float(line[1]))
                true_yaw = round(float(line[4]))

                if true_lat not in self.true_lat_list:
                    self.true_lat_list.append(true_lat)
                if true_lng not in self.true_lng_list:
                    self.true_lng_list.append(true_lng)
                if true_yaw not in self.true_yaw_list:
                    self.true_yaw_list.append(true_yaw)

                if true_roll not in self.true_roll_list:
                    self.true_roll_list.append(true_roll)
                if true_pitch not in self.true_pitch_list:
                    self.true_pitch_list.append(true_pitch)
                if true_alt not in self.true_alt_list:
                    self.true_alt_list.append(true_alt)

        print("true lat list is ", self.true_alt_list)
        self.open_file_w_headers_second(badfile, justname)

    # detour
    def open_file_w_headers_second(self, badfile, justname):
        false_counter = 0
        compared_file_list = []
        with open("../bad_auto/" + badfile) as f:
            next(f)

            for line in csv.reader(f):
                current_line = line
                print("alt is ", float(line[1]))
                bad_alt = format((float(line[1])))
                bad_lat = format(float(line[2]), '.5f')
                bad_lng = format(float(line[3]), '.5f')
                bad_yaw = round(float(line[4]))
                bad_roll = format(float(line[5]), '.1f')
                bad_pitch = format(float(line[6]), '.1f')

                # if bad_lat not in self.true_lat_list:
                #     false_counter += 1
                #     current_line.append(1)
                #     compared_file_list.append(current_line)
                #
                # elif bad_lng not in self.true_lng_list:
                #     false_counter += 1
                #     current_line.append(1)
                #     compared_file_list.append(current_line)

                if bad_alt not in self.true_alt_list:
                    false_counter += 1
                    current_line.append(1)
                    compared_file_list.append(current_line)

                # elif bad_roll not in self.true_roll_list:
                #     if bad_pitch not in self.true_pitch_list:
                #         if bad_yaw not in self.true_yaw_list:
                #                 false_counter += 1
                #                 current_line.append(1)
                #                 compared_file_list.append(current_line)
                else:
                    current_line.append(0)
                    compared_file_list.append(current_line)
        f.close()

        # print("false count: " + false_counter)
        # print("just name is " +  justname)
        os.chdir("../compared/")

        compared_file_name = justname + 'compared.csv'
        # print(compared_file_name)
        with open(compared_file_name, 'w', newline='') as openFile:
            # print('making compare file')
            writer = csv.writer(openFile)
            writer.writerow(first_line)
            for row in compared_file_list:
                writer.writerow(row)

        for row in compared_file_list:
            row.append(justname)
            full_data.append(row)
        openFile.close()

    # def check_category(self):
    #     cat_list = []
    #     with open('comparedNW.csv') as f:
    #         for line in csv.reader(f):
    #             print(line[6])
    #             if line[6] == '1':
    #                 cat_list.append(line[6])
    #     print('predicted number of incorrect values is', len(cat_list))
    #     print('actual number of incorrect values is 1096')


def main():
    obj = TestCsv()
    # obj.open_file_w_headers_second()

    goodFiles = []
    badFiles = []

    os.chdir("./formatted_auto")
    for file in glob.glob("*.csv"):
        goodFiles.append(file)

    os.chdir("../bad_auto")
    for file in glob.glob("*.csv"):
        badFiles.append(file)

    print(goodFiles)
    print(badFiles)

    # goodFiles.sort()
    # badFiles.sort()
    #
    count = 0
    for i in range(0, len(goodFiles)):
        # print(len(goodFiles))
        good_data_file_name = goodFiles[i]
        bad_data_file_name = badFiles[i]
        name = good_data_file_name.split('.')
        badname = bad_data_file_name.split('.')

        goodfilename = name[0].replace("gp", "")
        badfilename = badname[0].replace("dp", "")

        if goodfilename == badfilename:
            print("good_data_file_name is ", goodfilename, "bad_data_file_name ", badfilename[-3:])
            obj.open_file_w_headers(goodFiles[i], badFiles[i], badfilename[-3:])
        # count = count + 1

    # with open("all_data.csv", 'w', newline='') as openFile:
    #     print('making final file')
    #     writer = csv.writer(openFile)
    #     writer.writerow(first_line)
    #     for row in full_data:
    #         writer.writerow(row)


if __name__ == '__main__': main()

# 272
