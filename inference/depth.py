"Inference depth calulation"
import numpy as np
import cv2
from .config import HEIGHT_DB_FILE, _DEBUG

#_DEBUG = False

def nn_depth(depthoutfolder, unwrap, basecount):
    "Convert phase to height"
    #BASEFILE = HEIGHT_DB_FILE
    DBase = np.load(HEIGHT_DB_FILE)
    depth = np.zeros((160, 160), dtype=np.float64)
    zee=0
    for i in range(160): #adressing edge noise, can not be explained yet!!!!
        # print('i:', i)
        for j in range(160):
            s=0
            for s in range(0, basecount-1,1):
                if (unwrap[i,j]> DBase[i,j,s]):
                    ds = (unwrap[i,j] - DBase[i,j,s])/( DBase[i,j,s]- DBase[i,j,s-1])
                    zee = s+ds*1
                    break
                else:
                    s+=1
                    if s==basecount:
                        print('not found!')

            # print(i,j,unwrap[i,j],DBase[i,j,s])
            if zee == 0:
                print('not found')
            depth[i,j]= (zee/basecount*-20 + 35)*1
    im_depth = depth
    if _DEBUG:
        print('nndepthrange=', np.ptp(depth), np.max(depth), np.min(depth) )
        cv2.imwrite(str(depthoutfolder / 'depth.png'), im_depth)
    return im_depth
