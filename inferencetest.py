
"inferencetest"
import time
from pathlib import Path
from shutil import rmtree, copytree
from PIL import Image
from inference.process_input import process
from filter.masks import addmask_to_picture
from filter.pcl_utils import filter_f_area, filter_pcl
from filter.img_utils import change_brightness_contrast

_PERF = True
_FILTER = False

def process_testimage():
    "process i image folder"
    testfolder = Path(__file__).parent.parent / 'testdata/testtarget/render0'
    tmp_folder = Path(__file__).parent.parent / 'tmp'
    rmtree(tmp_folder, ignore_errors=True)
    copytree(testfolder, tmp_folder)
    myoutfolder = tmp_folder / 'out'
    myoutfolder.mkdir()
    if _PERF:
        proc_st = time.process_time()
        st_time = time.time()
    process(tmp_folder, myoutfolder)
    if _PERF:
        end_time = time.time()
        proc_end = time.process_time()
        print("CPU exec time:", proc_end-proc_st, "seconds")
        print("Elapsed time:", end_time-st_time, "seconds")
    print("Data processed")

def gen_image_set(folder):
    "genereate i test serie"
    tmp_folder = Path(__file__).parent / 'tmp'
    rmtree(tmp_folder, ignore_errors=True)
    tmp_folder.mkdir()
    for i in range(10):
        tmpdir = tmp_folder / ('render'+str(i))
        copytree(folder, tmpdir)
        #change_brightness_contrast(tmpdir/ 'image0.png', tmpdir / "image0.png", contrast=1, brightness=(2*i*0.2))
        #change_brightness_contrast(tmpdir/ 'image0.png', tmpdir / "image0.png", contrast=2*i*0.2, brightness=0.8)
        zoom
        process(tmpdir)
        #filter_pcl(tmp_folder / f.name / 'pointcloud.ply',tmp_folder / f.name / 'npointcloud.ply',0.2)

    
def process_image_set(folder):
    "process a image folder set"
    tmp_folder = Path(__file__).parent / 'tmp'
    rmtree(tmp_folder, ignore_errors=True)
    tmp_folder.mkdir()
    folders = sorted(folder.glob('*'))
    if _PERF:
        proc_st = time.process_time()
        st_time = time.time()
    for f in folders:
        if f.is_dir():
            print(f)
            copytree(f, tmp_folder / f.name)
            if _FILTER:
                fil=tmp_folder / f.name / 'image0.png'
                img = Image.open(fil)
                img = addmask_to_picture(img)
                img.save(fil)
                for fil in ['image8.png','image9.png']:
                    img = Image.open(tmp_folder / f.name / fil)
                    img=addmask_to_picture(img, maskval=0)
                    img.save(tmp_folder / f.name / fil)
            # myoutfolder = tmp_folder / 'out'
            # myoutfolder.mkdir()
            process(tmp_folder / f.name)
            filter_pcl(tmp_folder / f.name / 'pointcloud.ply',tmp_folder / f.name / 'npointcloud.ply',0.2)
    if _PERF:
        end_time = time.time()
        proc_end = time.process_time()
        print("CPU exec time:", proc_end-proc_st, "seconds")
        print("Elapsed time:", end_time-st_time, "seconds")
    print("Data processed")

if __name__=='__main__':
    #process_testimage()
    testset_folder = Path(__file__).parent.parent / 'testdata/testtarget'
    testset_folder = Path(__file__).parent.parent / 'testdata/1cm_target_220830'
    testset_folder = Path(__file__).parent / 'testdata/kugle_220906'
    #process_image_set(testset_folder)
    test_folder = Path(__file__).parent / 'testdata/input/render0'
    gen_image_set(test_folder)