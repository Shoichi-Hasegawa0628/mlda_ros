#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __init__ import *
import glob
import codecs
from std_msgs.msg import String

class HumanInput():
    def human_input_server(self, req):
        
        object_id = req.count
        
        if req.status == "no":
            word_file = open(DATA_FOLDER + "/word_dic.txt")
            word_dic = word_file.readlines()
            
            main_data = np.loadtxt(DATA_FOLDER + "/histogram_word.txt")
            
            if len(main_data) < object_id + 1: # Check the data existance
                add_data = np.loadtxt(DATA_FOLDER +  "/estimate_histogram_word" + str(object_id) + ".txt")
                main_data = np.append(main_data, add_data).reshape(-1, len(word_dic))
                np.savetxt(DATA_FOLDER + "/histogram_word.txt", main_data, fmt=str("%d"))
                rospy.loginfo("[Service em_mlda/human_input] Update the word histogram with estimated word directly")
            else:
                rospy.loginfo("[Service em_mlda/human_input] Finished the teaching this object")
                             
        elif req.status == "this":
            rospy.loginfo("[Service em_mlda/human_input] Teach this object as " + req.word)
            
            word_file = open(DATA_FOLDER + "/word_dic.txt")
            main_data = np.loadtxt(DATA_FOLDER + "/histogram_word.txt")
            
            word_dic = word_file.readlines()
            word_dic = [convert.replace("\n", "") for convert in word_dic]
            word_dic = [convert.replace(".", "") for convert in word_dic]
            
            if len(main_data) < object_id + 1: # Check the data existance
                teach_histogram = np.zeros(len(word_dic))      
            
                if req.word in word_dic:
                    index = word_dic.index(req.word)
                    teach_histogram[index] += 1
            
                main_data = np.append(main_data, teach_histogram).reshape(-1, len(word_dic))
                np.savetxt(DATA_FOLDER + "/histogram_word.txt", main_data, fmt=str("%d"))
                rospy.loginfo("[Service em_mlda/human_input] Update the word histogram with human input: %s", req.word)
            else:
                if req.word in word_dic:
                    index = word_dic.index(req.word)
                    main_data[object_id][index] += 1
                    
                np.savetxt(DATA_FOLDER + "/histogram_word.txt", main_data, fmt=str("%d"))
                rospy.loginfo("[Service em_mlda/human_input] Update the word count of %s:" + str(main_data[object_id][index]), req.word)                
            
        else:
            rospy.loginfo("[Service em_mlda/human_input] Known command %s", req.sentence)
        
        return mlda_human_inputResponse(True)
    
        
    def __init__(self):
        s = rospy.Service('em_mlda/human_input', mlda_human_input, self.human_input_server)
        rospy.loginfo("[Service em_mlda/human_input] Ready em_mlda/human_input")
        
        
if __name__ == "__main__":
    rospy.init_node('em_mlda_human_input_server')
    HumanInput()
    rospy.spin()
