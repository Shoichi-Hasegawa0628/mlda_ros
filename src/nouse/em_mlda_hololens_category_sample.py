#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __init__ import *
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

class MLDA_HoloLens_Category_Sample():
    def hololens_category_sample_server(self, req):
        if req.status == "visualize":
            resp = mlda_hololens_category_sampleResponse()

            Category_example_word = np.loadtxt(os.path.join(DATA_FOLDER + "/learn_result", "Category_example_word.txt"), delimiter=',', dtype="int64")
            # word_dic = np.loadtxt(os.path.join(DATA_FOLDER, "word_dic.txt"))
            word_file = open(os.path.join(DATA_FOLDER, "word_dic.txt"))
            word_dic = word_file.readlines()

            word_dic = [convert.replace("\n", "") for convert in word_dic]
            word_dic = [convert.replace(".", "") for convert in word_dic]

            print(word_dic)
            print(Category_example_word)

            for category in range(Category_example_word.shape[0]):
                for sample in range(Category_example_word.shape[1]):
                    resp.word_list.append(word_dic[Category_example_word[category][sample]])

            Category_example_image = np.loadtxt(os.path.join(DATA_FOLDER + "/learn_result", "Category_example_image.txt"), delimiter=',')

            for category in range(Category_example_image.shape[0]):
                for sample in range(Category_example_image.shape[1]):
                    image_path = DATA_FOLDER + "/image/" + str(int(Category_example_image[category][sample])) + ".jpg"
                    print(image_path)
                    im = cv2.imread(image_path, cv2.IMREAD_COLOR)

                    converted_image = CompressedImage()
                    converted_image.header.stamp = rospy.Time.now()
                    converted_image.format = "jpeg"
                    converted_image.data = np.array(cv2.imencode('.jpg', im)[1]).tostring()

                    resp.image_list.append(converted_image)

            image_path = DATA_FOLDER + "/image/" + str(int(req.object_num)) + ".jpg"
            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
            converted_image = CompressedImage()
            converted_image.header.stamp = rospy.Time.now()
            converted_image.format = "jpeg"
            converted_image.data = np.array(cv2.imencode('.jpg', im)[1]).tostring()
            resp.image_list.append(converted_image)

            return resp

        else:
            rospy.logwarn(
                "[Service em_mlda/hololens_category_sample] command not found: %s", req.status)

        return mlda_hololensResponse(True)

    def __init__(self):
        s = rospy.Service('em_mlda/hololens_category_sample', mlda_hololens_category_sample, self.hololens_category_sample_server)
        rospy.loginfo("[Service em_mlda/hololens_category_sample] Ready em_mlda/hololens_category_sample")

if __name__ == "__main__":
    rospy.init_node('em_mlda_hololens_category_sample_server')
    MLDA_HoloLens_Category_Sample()
    rospy.spin()
