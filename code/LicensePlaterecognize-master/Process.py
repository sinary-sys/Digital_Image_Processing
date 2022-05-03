###########################################################################################################

# Author: zackary Shen
# Email: szbltyy@hotmail.com
# Date: 2019/6/7 11:51 Fri

###########################################################################################################

# This is a class that can process image using OpenCV
# This class can apart the character from a license plate
# And show it to display

# tips: Comments annotated with # are comments for explanation
# tips: Comments annotated with ## are testing codes

import cv2


class PhotoPro:
    """
     This class can identify the character and color of lincense plate
    """

    # init the image, img just a String of path
    def __init__(self, image_path):
        self.color_img = image_path

    # To call the necessary function
    def Get_Image(self):
        self.Img_Read()
        img = self.Edge_Search()
        return img

    # To read the image
    def Img_Read(self):
        # To open a rgb image, because I need to record color
        self.color_img = cv2.imread(self.color_img)

        # Change img to a grey image
        self.gray_img = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2GRAY)
        ## cv2.imshow('Grey Figure', self.img)

    # Main process calling function to sign the area of license plate
    def Edge_Search(self):
        # To locate the license plate

        # using the cascade.xml to locate
        ###################################################################################

        # cascade.xml is Machine-based classifier based on Haar feature
        # , LBP feature, Hog feature; Can accurately identify the license plate
        # reference to the CSDN blog https://blog.csdn.net/lql0716/article/details/72566839

        ###################################################################################
        cascade_path = './data/cascade.xml'
        cascade = cv2.CascadeClassifier(cascade_path)  # classifier

        # To detect the area  most like the license plate in the original picture
        car_plates = cascade.detectMultiScale(self.color_img, 1.1, 2, \
                                              minSize=(36, 9), maxSize=(36 * 40, 9 * 40))

        if len(car_plates) > 0:
            for car_plate in car_plates:
                x, y, w, h = car_plate
                plate = self.color_img[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.rectangle(self.color_img, (x - 10, y - 10), (x + w + 10, \
                                                                 y + h + 10), (255, 0, 0), 2)

        ###################################################################################
        ## cv2.namedWindow("license", 0)
        ## cv2.resizeWindow("license", 640, 640)
        ## cv2.imshow("license", self.color_img)

        # This the license we want
        ## cv2.namedWindow("plate", 0)
        ## cv2.resizeWindow("plate", 640, 640)
        ## cv2.imshow("plate", plate)
        ## cv2.waitKey(0)
        ## cv2.destroyAllWindows()
        ###################################################################################
        # Return the license plate
        # And we choose the colorful plate, because we need to recognize the color, too
        return plate


# Test the class
if __name__ == '__main__':
    pp = PhotoPro('./Images/1.jpg')
    pp.Get_Image()
