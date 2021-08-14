#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __init__ import *
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import sys
import csv
import codecs


class GetJointLoadFeature():
    def joint_load_callback(self, joint_status):
        if self.save_mode:
            with open(DATA_FOLDER + "/joint" + str(self.current_joint_number) + "_load" + str(self.current_object_number) + ".csv", 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(joint_status.effort)

    def get_joint_load_feature(self, joint_num, object_num):
        fig = plt.figure(figsize=(15, 10), dpi=100)
        plt.rcParams["font.size"] = 20
        ax = fig.add_subplot(111)
        
        df = pd.read_csv(DATA_FOLDER + "/joint" + str(self.current_joint_number) + "_load" + str(self.current_object_number) + ".csv", header=None)
        list_df = df[joint_num - 1][:].values.tolist() * -1
        df_nda = df[joint_num - 1][:].values * -1

        if joint_num == 4:
            idx = signal.argrelmax(df_nda, order=1)
        else:
            idx = signal.argrelmin(df_nda, order=1)

        features = df_nda[idx]

        ax.plot(list_df, label="object " + str(object_num))
        ax.plot(idx[0], features, 'o', ms = 15, label="Max_peak_" + str(object_num))
        ax.legend()

        ax.set_title("Joint " +  str(joint_num), fontsize=20)
        ax.set_xlabel("Time", size=20, weight="light")
        ax.set_ylabel("Moment [N/m]", size=20, weight="light")
        plt.savefig(DATA_FOLDER + "/" + str(object_num) + "(" + str(joint_num) + ")" + ".png")

        if not ESTIMATE:
            with open(DATA_FOLDER + "/joint_load_feature_"+ str(joint_num) + ".csv", 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(features)
        else:
            with open(DATA_FOLDER + "/joint_load_feature_"+ str(joint_num) + ".csv", 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(features)
                
            with open(DATA_FOLDER + "/unknown" + str(object_num) + "_joint_load_feature"+ str(joint_num) + ".csv", 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(features)

        rospy.loginfo("[Service em_mlda/joint_load] Save the new joint load feature")
    
    def make_new_hist(self, object_num):
        with open(DATA_FOLDER + "/unknown" + str(object_num) + "_joint_load_feature8.csv") as f:
            self.features_hardness = np.array([row for row in csv.reader(f)])
        
        with open(DATA_FOLDER + "/unknown" + str(object_num) + "_joint_load_feature4.csv") as f:
            self.features_weight = np.array([row for row in csv.reader(f)])
        
        hist_hardness = np.zeros((len(self.features_hardness), len(self.dictionary_hardness)))
        hist_weight = np.zeros((len(self.features_weight), len(self.dictionary_weight)))
        
        for d, features in enumerate(self.features_hardness):
            for feature in features:
                rounded_feature = round(float(feature), 1)
                if rounded_feature in self.dictionary_hardness:
                    idx = self.dictionary_hardness.index(rounded_feature)
                    hist_hardness[d, idx] += 1

        for d, features in enumerate(self.features_weight):
            for feature in features:
                rounded_feature = round(float(feature), 1)
                if rounded_feature in self.dictionary_weight:
                    idx = self.dictionary_weight.index(rounded_feature)
                    hist_weight[d, idx] += 1
        
        np.savetxt(DATA_FOLDER + "/unknown_histogram_hardness" + str(object_num) + ".txt", hist_hardness, fmt=str("%d"))
        rospy.loginfo("[Service em_mlda/joint_load] save the histgram as " + "unknown_histogram_hardness" + str(object_num) + ".txt")

        np.savetxt(DATA_FOLDER + "/unknown_histogram_weight" + str(object_num) + ".txt", hist_weight, fmt=str("%d"))
        rospy.loginfo("[Service em_mlda/joint_load] save the histgram as " + "unknown_histogram_weight" + str(object_num) + ".txt")
            
        main_data = np.loadtxt(DATA_FOLDER +  "/histogram_hardness.txt")
        main_data = np.append(main_data, hist_hardness).reshape(-1, len(self.dictionary_hardness))
        np.savetxt(DATA_FOLDER + "/histogram_hardness.txt", main_data, fmt=str("%d"))

        main_data = np.loadtxt(DATA_FOLDER +  "/histogram_weight.txt")
        main_data = np.append(main_data, hist_weight).reshape(-1, len(self.dictionary_weight))
        np.savetxt(DATA_FOLDER + "/histogram_weight.txt", main_data, fmt=str("%d"))            

    def load_dic(self):
        self.dictionary_hardness = []
        self.dictionary_weight = []
        
        with open(DATA_FOLDER + "/dic_weight.txt") as f:
            for s_line in f:
                self.dictionary_weight.append(float(s_line))
        rospy.loginfo("[Service em_mlda/joint_load] Load the dictionary of weight")
                
        with open(DATA_FOLDER + "/dic_hardness.txt") as f:
            for s_line in f:
                self.dictionary_hardness.append(float(s_line))
        rospy.loginfo("[Service em_mlda/joint_load] Load the dictionary of hardness")

    def make_dic(self):
        with open(DATA_FOLDER + "/joint_load_feature_8.csv") as f:
            self.features_hardness = np.array([row for row in csv.reader(f)])
        
        with open(DATA_FOLDER + "/joint_load_feature_4.csv") as f:
            self.features_weight = np.array([row for row in csv.reader(f)])


        for each_object in self.features_hardness:
            for feature in each_object:
                rounded_feature = round(float(feature), 1)
                if rounded_feature not in self.dictionary_hardness:
                    self.dictionary_hardness.append(rounded_feature)
        
        for each_object in self.features_weight:
            for feature in each_object:
                rounded_feature = round(float(feature), 1)
                if rounded_feature not in self.dictionary_weight:
                    self.dictionary_weight.append(rounded_feature)
        
        with open(DATA_FOLDER + "/dic_weight.txt", 'w') as f:
            f.write('\n'.join(map(str, self.dictionary_weight)))
        rospy.loginfo("[Service em_mlda/joint_load] save dictionary as dic_weight.txt")

        with open(DATA_FOLDER + "/dic_hardness.txt", 'w') as f:
            f.write('\n'.join(map(str, self.dictionary_hardness)))
        rospy.loginfo("[Service em_mlda/joint_load] save dictionary as dic_hardness.txt")        

    def make_bof(self):
        hist_hardness = np.zeros((len(self.features_hardness), len(self.dictionary_hardness)))
        hist_weight = np.zeros((len(self.features_weight), len(self.dictionary_weight)))
        
        for d, features in enumerate(self.features_hardness):
            for feature in features:
                rounded_feature = round(float(feature), 1)
                idx = self.dictionary_hardness.index(rounded_feature)
                hist_hardness[d, idx] += 1

        for d, features in enumerate(self.features_weight):
            for feature in features:
                rounded_feature = round(float(feature), 1)
                idx = self.dictionary_weight.index(rounded_feature)
                hist_weight[d, idx] += 1
                
        np.savetxt(DATA_FOLDER + "/histogram_hardness.txt", hist_hardness, fmt=str("%d"))
        rospy.loginfo("[Service em_mlda/joint_load] save the histgram as histogram_hardness.txt")

        np.savetxt(DATA_FOLDER + "/histogram_weight.txt", hist_weight, fmt=str("%d"))
        rospy.loginfo("[Service em_mlda/joint_load] save the histgram as histogram_weight.txt")
    
    def joint_load_server(self, req):
        self.current_joint_number = req.joint_number
        self.current_object_number = req.count
        
        if req.sentence == "add":
            rospy.loginfo("[Service em_mlda/joint_load] Get the features of the joint %d", req.joint_number)
            self.get_joint_load_feature(req.joint_number, req.count)
        
        elif req.sentence == "record_on":
            rospy.loginfo("[Service em_mlda/joint_load] Start to the recording the joint load")
            self.save_mode = True
            
        elif req.sentence == "record_off":
            rospy.loginfo("[Service em_mlda/joint_load] Stop the recording")
            self.save_mode = False
            
        elif req.sentence == "learn":
            self.make_dic()
            self.make_bof()
            
        elif req.sentence == "estimate":
            main_data = np.loadtxt(DATA_FOLDER + "/histogram_weight.txt")
            if len(main_data) < req.count + 1: # Check the data existance
                self.load_dic()
                self.make_new_hist(req.count)
        else:
            rospy.logwarn("[Service em_mlda/data] command not found: %s", req.sentence)

        return mlda_joint_loadResponse(True)

    def __init__(self):
        rospy.Subscriber(JOINT_LOAD_TOPIC, JointState, self.joint_load_callback, queue_size=1)
        s = rospy.Service('em_mlda/data/joint_load', mlda_joint_load, self.joint_load_server)
        rospy.loginfo("[Service em_mlda/joint_load] Ready em_mlda/joint_load")

        self.features_hardness = []
        self.features_weight = []
        self.dictionary_hardness = []
        self.dictionary_weight = []
        self.save_mode = False
        self.current_object_number = 0
        self.current_joint_number = 8


if __name__ == "__main__":
    rospy.init_node('em_mlda_joint_load_server')
    GetJointLoadFeature()
    rospy.spin()
