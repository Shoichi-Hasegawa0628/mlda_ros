#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
import numpy as np
import random
import subprocess
import math

from mlda_ros.srv import *
from mlda_ros.msg import *

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#TRIALNAME = "Testno"
# TRIALNAME = "Testor"
TRIALNAME = "Testmr"
NEWDATA = False # When you make the dataset, you should write True
ESTIMATE = True # When you do the experiment, you should write True

ITERATION = 100
CATEGORYNUM = 6
CATEGORYEXAMPLENUM = 3
IMAGE_NUM = 4
ALPHA = 1.0
BETA = 1.0

IMAGE_TOPIC = "/sciurus17/camera/color/image_raw"
WORD_TOPIC="/sciurus17/em_mlda/word"
JOINT_LOAD_TOPIC="/sciurus17/controller2/joint_states"
JOINT_LOAD_SAVE_TOPIC="/sciurus17/em_mlda/joint_states_save"

DATASET_FOLDER = "../training_data/"
DATA_FOLDER = DATASET_FOLDER + TRIALNAME
