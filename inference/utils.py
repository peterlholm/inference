"inference utils"
import cv2
#import numpy as np

def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def resize( img, 
            H = 160,
            W = 160,

            plotstat = False,
            printstat = False,
            ):
    """
    Select fixed size square in the image
    TODO: is this needed? 
    """
    #printif(img.shape, printstat)
    img_resized = img[0:H,0:W]
    #printif(img_resized.shape, printstat)

    # if plotstat:
    #     plotListFig([img, img_resized], # list images
    #                 ["Original image", "Resized image"], # list titles
    #                 )

    return(img)
