###########################################################################################################

# Author: zackary Shen
# Email: szbltyy@hotmail.com
# Date: 2019/6/20 00:28 Thur

###########################################################################################################

# This is a class that recognize the characters from a photo
# The license plate will get from a object from class PhotoPro
# And then this class will recognize the number and color
# of the license plate, and store it in a String object

# tips: Comments annotated with # are comments for explanation
# tips: Comments annotated with ## are testing codes


import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import cv2
from Process import PhotoPro
from PIL import Image, ImageTk
import threading
import time
import predict

# predict is a recognize py file written by wzh from github
# homepage of wzh: https://github.com/wzh191920

class Recognizes:
    """
    This is a class that recognize the characters from a photo
    """

    # init the class
    def __init__(self, image_path):
        self.image_path = image_path

    # using a PhotoPro object to get plate
    def Get_Plate(self):
        # get the license plate, the color one need to recognize color
        photopro = PhotoPro(self.image_path)
        color_plate = photopro.Get_Image()

        # create a object from predict which can recognize plate
        cpd = predict.CardPredictor()
        cpd.train_svm()
        # write plate
        cv2.imwrite('test.jpg', color_plate)
        return cv2.imread('./test.jpg')

if __name__ == '__main__':
    pp = Recognizes('./Images/6.jpg')
    pp.Get_Plate()