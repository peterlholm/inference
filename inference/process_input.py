"Precess input folder"
import multiprocessing
import os
import time
from pathlib import Path
from shutil import rmtree, copytree
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # more logging
# disable cuda
#os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import tensorflow as tf
import cv2

from config import H_MODEL_FILE, L_MODEL_FILE
from utils import make_grayscale, resize
from masking import mask
from depth import nnDepth
from pointcloud import nngenerate_pointcloud

_DEBUG=False
_PERF = False

TF_VERBOSE=1    # progress bar
TF_VERBOSE=0    # silent
TF_VERBOSE=2

_NET2=True
_MASK=False


PI = 2*np.pi

# suppress tf tons of annoying messages:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

L_model = tf.keras.models.load_model(L_MODEL_FILE)
H_model = tf.keras.models.load_model(H_MODEL_FILE)

# def load_models():
#     global L_model, H_model
#     L_model = tf.keras.models.load_model(L_MODEL_FILE)
#     H_model = tf.keras.models.load_model(H_MODEL_FILE)

def db_kinfer(x):
    "run L-model"
    predicted_img = L_model.predict(np.array([np.expand_dims(x, -1)]),verbose=TF_VERBOSE)
    predicted_img = np.argmax(predicted_img, axis=-1)
    predicted_img = predicted_img.squeeze()
    return predicted_img

def process(folder):
    "Procss the folder with hole inference"
    if not folder.exists():
        raise Exception('Input folder dont exist')
    if _DEBUG:
        print("Starting processing", folder)
    #load_models()
    image0 = folder / 'image0.png'
    img = cv2.imread(str(image0)).astype(np.float32)
    img = resize(img, 160, 160)
    img = make_grayscale(img)
    if _DEBUG:
        cv2.imwrite(str(folder / 'gray.png'), img)
    inp_img = img/255
    mymask = mask(folder)
    inp_img = np.multiply(np.logical_not(mymask), inp_img)
    wrap_input = H_model.predict(np.array([np.expand_dims(inp_img, -1)]),verbose=TF_VERBOSE)
    wrap_input = wrap_input.squeeze()
    wrap_input = np.multiply(np.logical_not(mymask), wrap_input)
    if _DEBUG:
        cv2.imwrite(str(folder / 'wrapin.png'), 255*wrap_input)
    #print(inpfile)
    # mymask = mask(inFolder + '/render'+str(i)+'/')
    # inp_img = np.multiply(np.logical_not(mymask), inp_img)
    k_img = db_kinfer(wrap_input)
    unwrapdata = np.add(2*PI*wrap_input, np.multiply(2*PI,k_img) )
    if _DEBUG:
        cv2.imwrite(str(folder / 'unwrap.png'), unwrapdata)
        cv2.imwrite(str(folder / 'k.png'), k_img)
    nndepth =nnDepth(depthfolder=folder, unwrap= .50* unwrapdata,basecount= 50)
    nngenerate_pointcloud(folder / 'image8.png', folder / 'mask.png', nndepth, folder / 'pointcloud.ply')





if __name__=='__main__':
    testfolder = Path(__file__).parent.parent / 'testdata/testtarget/render0'
    tmp_folder = Path(__file__).parent.parent / 'tmp'
    rmtree(tmp_folder, ignore_errors=True)
    copytree(testfolder, tmp_folder)
    if _PERF:
        proc_st = time.process_time()
        st_time = time.time()

    process(tmp_folder)
    if _PERF:
        end_time = time.time()
        proc_end = time.process_time()
        print("CPU exec time:", proc_end-proc_st, "seconds")
        print("Elapsed time:", end_time-st_time, "seconds")

    print("Data processed")
