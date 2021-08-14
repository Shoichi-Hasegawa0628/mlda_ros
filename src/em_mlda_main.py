#!/usr/bin/env python
from __init__ import *

if __name__ == "__main__":
    if NEWDATA:
        print("NEWDATA MODE START")
        subprocess.Popen("mkdir -p " + DATA_FOLDER, shell=True)
        subprocess.Popen("mkdir -p " + DATA_FOLDER + "/image", shell=True)
        subprocess.Popen("rm " + DATA_FOLDER + "/codebook.txt", shell=True)
        subprocess.Popen("rm " + DATA_FOLDER + "/histogram_image.txt", shell=True)
        subprocess.Popen("rm " + DATA_FOLDER + "/word_dic.txt", shell=True)
        subprocess.Popen("rm " + DATA_FOLDER + "/word.txt", shell=True)
        subprocess.Popen("rm " + DATA_FOLDER + "/histogram_word.txt", shell=True)
        subprocess.Popen("rm " + DATA_FOLDER + "/joint_load*.csv", shell=True)