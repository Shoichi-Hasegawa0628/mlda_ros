#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MLDAのコードを統括するコード

import rospy
from mlda_ros.srv import SendImageMLDA
from mlda_ros.srv import SendImageMLDAResponse
import mlda_ros_data_image
import mlda_ros_learn
import mlda_ros_data_word


class MLDAMain():
    
    def __init__(self):
        self.mlda_image = mlda_ros_data_image.GetImageFeature()
        self.mlda_learn = mlda_ros_learn.MLDA()
        self.mlda_word = mlda_ros_data_word.GetWordFeature()
        rospy.Service('judge_mlda', SendImageMLDA, self.judge_target_object_mlda)
    

    def judge_target_object_mlda(self, msg):
        if msg.status == "estimate":                                                        # 物体の画像から物体の単語を予測
            self.mlda_image.image_server(msg)
            self.mlda_learn.mlda_server(msg)
            object_word, object_word_probability = self.mlda_word.word_server(msg)
 
        else:                                                                               # データセットからパラメータを学習
            pass
        
        return SendImageMLDAResponse(success = True, object_word, object_word_probability)


if __name__ == "__main__":
    rospy.init_node('mlda_main_server')
    MLDAMain()
    rospy.spin()