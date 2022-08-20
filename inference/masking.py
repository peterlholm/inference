"Inference masking"
#from pathlib import Path
import numpy as np
import cv2
from config import HEIGHT, WIDTH
from utils import make_grayscale

_DEBUG = False

def mask(folder):
    "Create the inference masing files"
    color = folder / 'image8.png'
    print('color:', color)
    img1 = np.zeros((HEIGHT, WIDTH), dtype=np.float)
    img1 = cv2.imread(str(color), 1).astype(np.float32)
    gray = make_grayscale(img1)
    black = folder / 'image9.png'
    img2 = np.zeros((HEIGHT, WIDTH), dtype=np.float)
    img2 = cv2.imread(str(black), 0).astype(np.float32)
    diff1 = np.subtract(gray, .5*img2)
    mymask =  np.zeros((HEIGHT, WIDTH), dtype=np.float)
    if _DEBUG:
        cv2.imwrite(str(folder / 'diff_mask.png'), diff1)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if (diff1[i,j]<40):
                mymask[i,j]= True
    if _DEBUG:
        np.save( folder / 'mask.npy', mymask, allow_pickle=False)
    cv2.imwrite( str(folder / 'mask.png'), 255*mymask)
    return(mymask)
