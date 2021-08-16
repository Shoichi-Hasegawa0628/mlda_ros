#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __init__ import *
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import glob
import pickle

class GetImageFeature():
    def calc_feature(self, filename):
        img = cv2.imread(filename, 0)
        kp, discriptor = self.detector.detectAndCompute(img, None)
        extraceted_img = cv2.drawKeypoints(img, kp, None, flags=4)
        # cv2.imwrite(filename.replace('.jpg', '') + "ex.jpg",extraceted_img)
        return np.array(discriptor, dtype=np.float32)


    def make_codebook(self, images, code_book_size, save_name):
        bow_trainer = cv2.BOWKMeansTrainer(code_book_size)

        for img in images:
            f = self.calc_feature(img)
            bow_trainer.add(f)

        code_book = bow_trainer.cluster()
        np.savetxt(DATA_FOLDER + "/" + save_name, code_book)


    def make_bof(self, code_book_name, images, hist_name, estimate_mode, objectID):
        code_book = np.loadtxt(DATA_FOLDER + "/" + code_book_name, dtype=np.float32)
        self.knn.train(code_book, cv2.ml.ROW_SAMPLE, np.arange(len(code_book), dtype=np.float32))

        hists = []
        
        counter = 0
        h = np.zeros(len(code_book))

        for img in images:
            f = self.calc_feature(img)
            idx = self.knn.findNearest(f, 1)[1]
            
            for i in idx:
                h[int(i)] += 1

            counter += 1
            
            if counter == IMAGE_NUM:            # counterがIMAGE_NUM (4枚(1物体につき))と等しいとき
                hists.append(h)                 # histsの配列に、
                h = np.zeros(len(code_book))
                counter = 0
        
        if estimate_mode:
            main_data = np.loadtxt(DATA_FOLDER + "/histogram_image.txt")
            main_data = np.append(main_data, hists).reshape(-1, 50)
            
            np.savetxt(DATA_FOLDER + "/histogram_image.txt", main_data, fmt=str("%d"))
            np.savetxt(DATA_FOLDER + "/" + hist_name, hists, fmt=str("%d"))
            
            subprocess.Popen("cp " + DATA_FOLDER + "/test_image/" + str(objectID) + "_*.jpg " + DATA_FOLDER + "/image", shell=True)
            
        else:
            np.savetxt(DATA_FOLDER + "/histogram_image.txt", hists, fmt=str("%d"))
            
            
    def image_server(self, req):
        if req.status == "learn":
            files = glob.glob(DATA_FOLDER + "/image/" + "*.jpg")
            self.make_codebook(files, 50, "codebook.txt")
            self.make_bof("codebook.txt", files, "histogram_image.txt", False, req.count)
        elif req.status == "estimate":
            main_data = np.loadtxt(DATA_FOLDER + "/histogram_image.txt")
            if len(main_data) < req.count + 1: # Check the data existance
                files = glob.glob(DATA_FOLDER + "/test_image/" + str(req.count) + "*.jpg")
                self.make_bof("codebook.txt", files, "unknown" + "_histogram_image" + str(req.count) + ".txt", True, req.count)
        else:
            if self.image_counter == IMAGE_NUM:
                self.image_counter = 0
            count = req.count
            image_name = DATA_FOLDER + "/image/" + str(count) + "_" + str(self.image_counter) + ".jpg"
            self.image_counter += 1
            cv2.imwrite(image_name, self.bridged_image)
            rospy.loginfo("[Service em_mlda/image] save new image as %s", image_name)

        return mlda_imageResponse(True)


    def image_callback(self, image):
        bridge = CvBridge()
        try:
            self.bridged_image = bridge.imgmsg_to_cv2(image, "bgr8")[226:480,371:640]
        except CvBridgeError as e:
            print (e)


    def __init__(self):
        rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        s = rospy.Service('em_mlda/data/image', mlda_image, self.image_server)
        
        self.knn = cv2.ml.KNearest_create()
        self.detector = cv2.AKAZE_create()
        # self.detector = cv2.KAZE_create()
        self.image_counter = 0
        
        
        
        rospy.loginfo("[Service em_mlda/image] Ready em_mlda/image")


if __name__ == "__main__":
    rospy.init_node('em_mlda_image_server')
    GetImageFeature()
    rospy.spin()
