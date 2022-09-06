"Simple image manipulations"

from PIL import Image, ImageEnhance
import cv2
import numpy as np

def rotate_img(infile, degree, outfile):
    img = Image.open(infile)
    new = img.rotate(degree, resample=Image.BILINEAR) #resample=Image.BILINEAR
    new.save(outfile)

def change_contrast(infile, outfile, value = 1.5):
    img = Image.open(infile)
    newimage = ImageEnhance.Contrast(img).enhance(value)
    newimage.save(outfile)

def change_brightness(infile, outfile, value= 1.5):
    img = Image.open(infile)
    newimage = ImageEnhance.Brightness(img).enhance(value)
    newimage.save(outfile)

def change_contrast_brightness(infile, outfile, contrast=1, brightness=1):
    img = Image.open(infile)
    image1 = ImageEnhance.Contrast(img).enhance(contrast)
    image2 = ImageEnhance.Brightness(image1).enhance(brightness)
    image2.save(outfile)

def change_brightness_contrast(infile, outfile, contrast=1, brightness=1):
    img = Image.open(infile)
    image1 = ImageEnhance.Brightness(img).enhance(brightness)
    image2 = ImageEnhance.Contrast(image1).enhance(contrast)
    image2.save(outfile)

def convert_to_grey(infile, outfile):
    img = Image.open(infile)
    grey = img.convert('L')
    grey.save(outfile)

def convert_green_to_grey(infile, outfile):
    img = Image.open(infile)
    green = img.getchannel(1)
    green.save(outfile)

def make_grayscale2(infile, outfile):
    # Transform color image to grayscale
    image = cv2.imread(str(infile))
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(outfile), gray_img)

def zoom(infile,outfile,zoom):
    img = Image.open(infile)
    newimg = img.resize((int(img.height*zoom)), int(img.width*zoom))
    x = int(img.width*zoom)-
    new2img = newimg.crop()
